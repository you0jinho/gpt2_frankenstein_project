# %% [markdown]
# # GPT 2.0 (Full version) — Frankenstein + 실제 GPT-2 학습 기법 6가지
#
# 이 파일은 notebook_06의 `TinyGPT` 구조를 베이스로, 실제 GPT-2가 사용하는
# 검증된 학습 기법 6가지를 추가한 버전입니다.
#
#   1. Weight tying (token embedding ↔ 출력층 가중치 공유)
#   2. GELU 활성화 함수 (FeedForward의 ReLU 대체)
#   3. Learning rate warmup + cosine decay
#   4. AdamW decoupled weight decay (bias/LayerNorm은 decay 제외, nanoGPT 방식)
#   5. Gradient clipping
#   6. Train/Val split + top-k/temperature sampling
#
# 데이터셋은 Tiny Shakespeare 대신 *Frankenstein* (Mary Shelley, 1818,
# Project Gutenberg #84, 퍼블릭 도메인)을 사용합니다.
#
# 실행 방법: VSCode + Python/Jupyter 확장이 있으면 `# %%` 위에 뜨는
# "Run Cell" 버튼으로 셀 단위 실행. 또는 터미널에서
# `python "real_gpt2.0.py"` 로 통째로 실행해도 됩니다.

# %%
import math
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# %% [markdown]
# ## 0. 데이터 다운로드 & 정제
#
# Project Gutenberg 텍스트는 앞뒤에 라이선스 안내문이 붙어 있어서, 실제
# 본문만 잘라냅니다.

# %%
DATA_PATH = Path("frankenstein.txt")
# 미러 두 개를 순서대로 시도합니다 (한쪽이 막혀 있을 수 있어서).
GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
]


def download_text(path: Path, urls) -> None:
    if path.exists():
        return
    last_error = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, path)
            return
        except Exception as e:  # noqa: BLE001 - 다음 미러로 넘어가기 위해 일단 잡음
            last_error = e
    raise RuntimeError(
        f"모든 다운로드 URL이 실패했습니다: {urls}\n"
        "인터넷이 막혀 있다면, 브라우저로 위 URL 중 하나를 열어 텍스트를 복사해서 "
        f"이 스크립트와 같은 폴더에 '{path.name}' 파일로 직접 저장하세요."
    ) from last_error


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Project Gutenberg 텍스트 앞뒤의 라이선스/안내문을 제거하고 본문만 반환."""
    import re

    start_match = re.search(r"\*\*\*\s*START OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw_text)
    end_match = re.search(r"\*\*\*\s*END OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw_text)

    start_idx = 0
    if start_match:
        start_idx = raw_text.find("\n", start_match.end()) + 1

    end_idx = len(raw_text)
    if end_match:
        end_idx = end_match.start()

    cleaned = raw_text[start_idx:end_idx].strip()
    return cleaned if cleaned else raw_text.strip()


download_text(DATA_PATH, GUTENBERG_URLS)
raw_text = DATA_PATH.read_text(encoding="utf-8")
text = strip_gutenberg_boilerplate(raw_text)

print("raw length:", len(raw_text))
print("cleaned length:", len(text))
print(text[:300])

# %% [markdown]
# ## 1. Vocab + Train/Val split

# %%
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

print("vocab_size:", vocab_size)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print("train tokens:", len(train_data), "| val tokens:", len(val_data))


class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


block_size = 256

train_dataset = NextTokenDataset(train_data, block_size)
val_dataset = NextTokenDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

xb, yb = next(iter(train_loader))
print("xb.shape:", xb.shape, "| yb.shape:", yb.shape)

# %% [markdown]
# ## 2. Model — notebook_06 구조 + GPT-2 스타일 개선 (#1, #2)
#
# `Head`/`MultiHeadAttention`/`Block`은 구조적으로 notebook_06과 동일합니다.
# 바뀐 부분: `FeedForward`의 활성화 함수(GELU), 모델 전체의 weight tying.

# %%
class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.2):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """GPT-2는 ReLU가 아니라 GELU를 사용합니다. (기법 #2)"""

    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT2Mini(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=256, num_heads=8, num_layers=6, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

        # --- 기법 #1: weight tying ---
        # token embedding과 출력층(lm_head)이 같은 가중치를 공유합니다.
        # 둘 다 shape가 (vocab_size, emb_dim)으로 동일하기 때문에 그대로 공유 가능합니다.
        # 실제 GPT-2도 이 트릭을 사용합니다 (파라미터 수 절약 + 약한 정규화 효과).
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)[None]
        h = self.drop(tok + pos_emb)
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), targets)
        return logits, loss

    def configure_optimizer(self, lr, weight_decay):
        """기법 #4: decoupled weight decay (nanoGPT 방식).

        2차원 이상 파라미터(Linear/Embedding weight 행렬)에만 weight decay를 걸고,
        bias와 LayerNorm 파라미터(1차원)는 decay 대상에서 제외합니다.
        """
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """기법 #6: temperature + top-k sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        self.train()
        return idx


model_preview = GPT2Mini(vocab_size, block_size)
n_params = sum(p.numel() for p in model_preview.parameters())
print(f"model parameters: {n_params/1e6:.2f}M")
del model_preview

# %% [markdown]
# ## 3. Learning rate: warmup + cosine decay (기법 #3)
#
# 처음 `warmup_iters` 동안 lr을 0에서 `max_lr`까지 선형으로 올리고, 그 뒤에는
# cosine 곡선을 따라 `min_lr`까지 천천히 낮춥니다.

# %%
def get_lr(it, warmup_iters, lr_decay_iters, min_lr, max_lr):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# %% [markdown]
# ## 4. 학습 (기법 #4, #5 포함)
#
# `device`는 NVIDIA GPU가 있으면 `cuda`, Apple Silicon Mac이면 `mps`, 둘 다
# 없으면 `cpu`로 자동 선택됩니다.
#
# `max_iters`가 너무 오래 걸리면 줄이세요 (GPU 없으면 500~1000 추천).

# %%
@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters=20):
    model.eval()
    losses = []
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
        if i + 1 >= eval_iters:
            break
    model.train()
    return sum(losses) / len(losses)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()
print("device:", device)

max_lr = 3e-4
min_lr = 3e-5
warmup_iters = 100
max_iters = 3000  # GPU 없으면 500~1000 정도로 줄이세요.
lr_decay_iters = max_iters
weight_decay = 0.1
eval_interval = 200

model = GPT2Mini(vocab_size, block_size).to(device)
optimizer = model.configure_optimizer(max_lr, weight_decay)

train_losses, val_losses, logged_steps = [], [], []


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


train_iter = cycle(train_loader)

try:
    for it in range(max_iters):
        lr = get_lr(it, warmup_iters, lr_decay_iters, min_lr, max_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        xb, yb = next(train_iter)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        # --- 기법 #5: gradient clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % eval_interval == 0 or it == max_iters - 1:
            tr = estimate_loss(model, train_loader, device)
            va = estimate_loss(model, val_loader, device)
            train_losses.append(tr)
            val_losses.append(va)
            logged_steps.append(it)
            print(f"step {it:5d} | lr {lr:.2e} | train loss {tr:.4f} | val loss {va:.4f}")
except KeyboardInterrupt:
    print("학습을 중간에 멈췄습니다. 지금까지의 결과로 계속 진행합니다.")

if val_losses:
    final_val_loss = val_losses[-1]
    print(f"final val loss: {final_val_loss:.4f} | val perplexity: {math.exp(final_val_loss):.2f}")

# %% [markdown]
# ## 5. Loss curve 저장

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(logged_steps, train_losses, label="train loss")
plt.plot(logged_steps, val_losses, label="val loss")
plt.xlabel("step")
plt.ylabel("cross entropy loss")
plt.title("GPT 2.0 (Full) training curve — Frankenstein, char-level")
plt.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
print("saved loss_curve.png")
try:
    plt.show()
except Exception:
    pass

# %% [markdown]
# ## 6. Sampling (기법 #6: temperature + top-k)

# %%
@torch.no_grad()
def generate_from_prompt(model, prompt, stoi, itos, device, max_new_tokens=500,
                          temperature=0.8, top_k=40):
    ids = [stoi[ch] for ch in prompt if ch in stoi]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    return "".join(itos[i] for i in out_ids)


prompt = "It was on a dreary night of November"
print(generate_from_prompt(model, prompt, stoi, itos, device, max_new_tokens=500, temperature=0.8, top_k=40))

torch.save(model.state_dict(), "gpt2_mini_frankenstein.pt")
print("saved gpt2_mini_frankenstein.pt")

# %% [markdown]
# ## 7. 정리
#
# | | notebook 6 (TinyGPT) | GPT 2.0 (Full) |
# |---|---|---|
# | 데이터셋 | Tiny Shakespeare | Frankenstein |
# | Train/Val split | 없음 | 90/10 |
# | Activation | ReLU | GELU |
# | 출력층 | 별도 가중치 | token embedding과 weight tying |
# | Learning rate | 고정 | warmup + cosine decay |
# | Weight decay | 전체 동일 적용 | 행렬에만 적용 (bias/LN 제외) |
# | Gradient clipping | 없음 | 있음 (max_norm=1.0) |
# | Sampling | 순수 multinomial | temperature + top-k |
#
# 구조(멀티헤드 어텐션, residual, layernorm, block 쌓기)는 notebook 6과
# 동일하게 유지하고, 학습 레시피와 데이터셋만 실제 GPT-2 수준으로 끌어올린
# 버전입니다.
