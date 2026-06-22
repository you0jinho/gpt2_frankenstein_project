# %% [markdown]
# # Notebook 7 — GPT 2.0 (Frankenstein Edition)
#
# **이 노트북에서 바뀌는 것은 딱 두 가지뿐입니다.**
#
# 1. **데이터셋**: Tiny Shakespeare → *Frankenstein* (Mary Shelley, 1818,
#    Project Gutenberg #84, 퍼블릭 도메인)
# 2. **Train / Validation split 추가**: notebook_06은 train loss만 봤습니다.
#    여기서는 데이터의 10%를 떼어내 validation으로 쓰고, train loss와 val
#    loss를 같이 추적해서 "외운 것"과 "일반화된 것"을 구분할 수 있는 결과를
#    보여줍니다.
#
# `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `TinyGPT` 클래스는
# **notebook_06과 글자 하나도 다르지 않습니다.** 구조를 바꾸지 않은 이유는,
# GPT 2.0의 목표가 "더 복잡한 모델을 만드는 것"이 아니라 "같은 GPT 구조를
# 다른 데이터셋으로 학습시키고 그 결과를 보여주는 것"이기 때문입니다.
#
# 실행 방법: VSCode에 Python + Jupyter 확장이 있으면 `# %%` 위에 뜨는
# **"Run Cell"** 버튼으로 한 셀씩 실행할 수 있습니다. 또는 터미널에서
# `python notebook_07_gpt2.py`로 통째로 실행해도 됩니다.

# %%
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# %% [markdown]
# ## 0. 데이터 다운로드 & 정제
#
# notebook_06은 이미 깨끗한 `input.txt`(Tiny Shakespeare)를 그대로 받아서
# 썼습니다. Project Gutenberg 파일은 책 본문 앞뒤에 라이선스 안내문이
# 붙어있어서, 그 부분을 잘라내는 전처리가 한 단계 더 필요합니다.
# (이 부분이 notebook_06에는 없던 유일한 "데이터 관련" 코드입니다.)

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
    """Project Gutenberg 텍스트 앞뒤의 라이선스 안내문을 제거하고 본문만 반환.

    Gutenberg 파일은 "*** START OF THE PROJECT GUTENBERG EBOOK ***"와
    "*** END OF THE PROJECT GUTENBERG EBOOK ***" 사이에 실제 책 내용이 있습니다.
    이 마커를 못 찾으면(파일 형식이 다르면) 원문 전체를 그대로 사용합니다.
    """
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

print("raw length (boilerplate 포함):", len(raw_text))
print("cleaned length (본문만):", len(text))
print(text[:300])

# %% [markdown]
# ## 1. Vocab + Dataset — notebook_06과 동일한 구조 + train/val split만 추가
#
# `NextTokenDataset`은 notebook_04~06과 완전히 같습니다. 추가된 부분은
# 데이터를 90%/10%로 나누는 두 줄뿐입니다.

# %%
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

print("vocab_size:", vocab_size)

# --- notebook_06에는 없던 부분: train/val split ---
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print("train tokens:", len(train_data), "| val tokens:", len(val_data))


class NextTokenDataset(Dataset):
    """notebook_04~06과 완전히 동일."""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


block_size = 64  # notebook_06과 동일

train_dataset = NextTokenDataset(train_data, block_size)
val_dataset = NextTokenDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

xb, yb = next(iter(train_loader))
print("xb.shape:", xb.shape, "| yb.shape:", yb.shape)

# %% [markdown]
# ## 2. Model — notebook_06의 `TinyGPT`와 글자 단위로 동일
#
# `Head` (masked self-attention) → `MultiHeadAttention` → `FeedForward` →
# `Block` (residual + layernorm) → `TinyGPT` (block 쌓기). 전부 notebook_06에서
# 그대로 가져왔습니다.

# %%
class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
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
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]
        h = tok + pos
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


model_preview = TinyGPT(vocab_size, block_size)
n_params = sum(p.numel() for p in model_preview.parameters())
print(f"model parameters: {n_params/1e6:.2f}M")
del model_preview

# %% [markdown]
# ## 3. 학습 — notebook_06의 `train_one_epoch`과 동일 + `evaluate` 함수 추가
#
# `evaluate`는 `train_one_epoch`에서 `optimizer.zero_grad()` /
# `loss.backward()` / `optimizer.step()` 세 줄만 빼고 `@torch.no_grad()`를
# 붙인 것입니다. 즉 "학습"과 "평가"의 코드 차이는 딱 그 세 줄뿐입니다.
#
# **속도 조절은 여기서 합니다.** 너무 오래 걸리면 `NUM_EPOCHS`나
# `MAX_STEPS_PER_EPOCH`를 줄이세요. (자세한 설명은 README 참고)

# %%
def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)


def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    """notebook_06과 완전히 동일."""
    model.train()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, device, max_steps=None):
    """train_one_epoch과 같은 구조이지만 backward/step이 없습니다 (= 학습 안 함)."""
    model.eval()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    model.train()
    return total_loss / total_count


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# --- 속도/시간을 조절하는 4가지 숫자 ---------------------------------------
NUM_EPOCHS = 15          # notebook_06은 100. 시간 부족하면 5~10으로.
MAX_STEPS_PER_EPOCH = 200  # notebook_06은 300. 시간 부족하면 50~100으로.
# ----------------------------------------------------------------------

model = TinyGPT(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_history, val_history = [], []
start_time = time.time()

try:
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, max_steps=MAX_STEPS_PER_EPOCH)
        val_loss = evaluate(model, val_loader, device, max_steps=50)
        train_history.append(train_loss)
        val_history.append(val_loss)
        elapsed = time.time() - start_time
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {elapsed:5.1f}s elapsed")
except KeyboardInterrupt:
    # Ctrl+C로 중간에 멈춰도, 지금까지 학습된 결과로 그래프/생성까지 마무리합니다.
    print("학습을 중간에 멈췄습니다. 지금까지의 결과로 계속 진행합니다.")

# %% [markdown]
# ## 4. 결과 — Loss curve (이게 바로 "다른 데이터셋으로 학습시킨 결과")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(train_history, label="train loss")
plt.plot(val_history, label="val loss")
plt.xlabel("epoch")
plt.ylabel("cross entropy loss")
plt.title("GPT 2.0 (Frankenstein) — train vs val loss")
plt.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
print("saved loss_curve.png")
try:
    plt.show()
except Exception:
    pass

# %% [markdown]
# ## 5. Sampling — notebook_06의 `sample_gpt`와 동일

# %%
@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, device, start_text="It was on a dreary night", max_new_tokens=400):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)
    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    model.train()
    return "".join(out)


print(sample_gpt(model, block_size, stoi, itos, device, start_text="It was on a dreary night", max_new_tokens=400))

torch.save(model.state_dict(), "tinygpt_frankenstein.pt")
print("saved tinygpt_frankenstein.pt")

# %% [markdown]
# ## 6. 정리
#
# | | notebook_06 (Tiny Shakespeare) | notebook_07 (GPT 2.0) |
# |---|---|---|
# | 모델 구조 | `TinyGPT` | **동일** (한 글자도 안 바뀜) |
# | 데이터셋 | Tiny Shakespeare | Frankenstein |
# | 평가 | train loss만 출력 | train/val loss 둘 다 추적 + 그래프 |
#
# 구조를 똑같이 유지한 이유는, "GPT 2.0"의 핵심 과제가 더 복잡한 모델을
# 설계하는 것이 아니라 **같은 GPT를 다른 데이터셋으로 학습시키고 결과를
# 검증하는 것**이기 때문입니다. 코드 한 줄 한 줄에 대한 자세한 설명은
# README.md에 정리해 두었습니다.
