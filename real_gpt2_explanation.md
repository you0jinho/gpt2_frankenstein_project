# `real_gpt2.0.py` 코드 상세 설명

# GPT2.0 의 여러요소들을 구현해보았습니다. 제출과제는 [`notebook_07_gpt2.py`](./notebook_07_gpt2.py) 이지만 참고하여 주시면 감사하겠습니다.
> 'real_gpt2.0.py' 의 실행결과는  [`real_gpt2_result.md`](./real_gpt2_result.md)에서 확인해주시면 감사하겠습니다.

---

## 0. 전체 그림

이 파일은 notebook_06의 `TinyGPT`(멀티헤드 self-attention + residual +
layernorm을 쌓은 구조)를 그대로 베이스로 삼고, 두 가지를 바꿉니다.

1. **데이터셋**: Tiny Shakespeare → *Frankenstein* (Mary Shelley, 1818,
   Project Gutenberg #84, 퍼블릭 도메인)
2. **학습 방식**: 실제 GPT-2가 쓰는 검증된 학습 기법 6가지를 추가
   - ① Weight tying
   - ② GELU 활성화 함수
   - ③ Learning rate warmup + cosine decay
   - ④ Decoupled weight decay (nanoGPT 방식)
   - ⑤ Gradient clipping
   - ⑥ Train/Val split + temperature/top-k sampling

구조(attention 자체)는 안 바꿨고, **"같은 구조를 더 제대로/안정적으로
학습시키는 법"** 에 초점을 맞춘 버전입니다.

---

## 1. Imports

```python
import math
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

- `math`: learning rate를 cosine 곡선으로 계산할 때 `math.cos`, `math.pi`가 필요합니다.
- `urllib.request`: 인터넷에서 텍스트 파일을 다운로드할 때 사용 (Colab의
  `!wget`과 같은 역할을 순수 Python으로 한 것 — 어디서든 동일하게 동작).
- `torch.nn`: 신경망 레이어(`Linear`, `Embedding`, `LayerNorm` 등)
- `torch.nn.functional as F`: `softmax`, `cross_entropy`, `gelu` 같은
  함수형 연산
- `Dataset`, `DataLoader`: PyTorch의 데이터 적재 인터페이스. notebook 4~6과 동일.

---

## 2. 데이터 다운로드 & 정제

### 2-1. `download_text`

```python
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
        except Exception as e:
            last_error = e
    raise RuntimeError(...) from last_error
```

- `if path.exists(): return` — 이미 `frankenstein.txt`가 있으면 또 받지
  않습니다. (매번 실행할 때마다 인터넷에서 다시 받으면 느리고, 네트워크가
  막혀 있으면 실행 자체가 안 되니까요.)
- `urllib.request.urlretrieve(url, path)` — URL에서 파일을 받아 `path`에
  그대로 저장합니다.
- URL을 **두 개** 등록해놓고 `for url in urls`로 순서대로 시도합니다. 첫
  번째가 실패(`Exception`)하면 두 번째를 시도하고, 둘 다 실패하면
  `RuntimeError`를 발생시켜서 "왜 실패했는지 + 대안(직접 다운로드해서
  저장)"을 안내합니다. → 한쪽 미러 서버가 막혀 있어도 동작하게 만든
  방어적 코드입니다.

### 2-2. `strip_gutenberg_boilerplate`

```python
def strip_gutenberg_boilerplate(raw_text: str) -> str:
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
```

- Project Gutenberg가 배포하는 파일은 책 본문 앞뒤에 "이 책은 퍼블릭
  도메인입니다, 이용 약관은..." 같은 법적 안내문이 붙어 있습니다. 이걸 안
  지우면 모델이 소설 문체가 아니라 안내문 문체까지 학습하게 됩니다.
- `re.search(패턴, 문자열)`: 정규식으로 `*** START OF THE/THIS PROJECT
  GUTENBERG EBOOK ... ***` 형태의 마커를 찾습니다. `TH(IS|E)`로
  "THIS"/"THE" 두 표기 방식(파일 버전에 따라 다름)을 모두 처리합니다.
  `.*?`는 "가능한 한 적게 일치"(non-greedy)하는 와일드카드로, 줄 끝
  `***`까지만 정확히 잘라냅니다.
- `start_match.end()` 다음에 오는 첫 줄바꿈(`\n`) 다음부터를 본문
  시작점으로 잡습니다 (마커가 있는 줄 자체는 본문이 아니므로 건너뜀).
- 마커를 못 찾으면(`if start_match:`가 거짓) `start_idx = 0`, 즉 처음부터
  전체를 다 사용합니다 — **fallback**: 정규식이 안 맞아도 프로그램이
  죄다 비어버리지 않게 하는 안전장치입니다.
- 마지막 줄 `cleaned if cleaned else raw_text.strip()`도 같은 이유의
  안전장치입니다.

### 2-3. 실제 호출부

```python
download_text(DATA_PATH, GUTENBERG_URLS)
raw_text = DATA_PATH.read_text(encoding="utf-8")
text = strip_gutenberg_boilerplate(raw_text)
```

다운로드 → 파일 읽기 → 정제, 3단계가 순서대로 실행됩니다. `text`가 이후
모든 단계에서 사용할 "순수한 소설 본문"입니다.

---

## 3. Vocab + Train/Val Split

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
```

- `set(text)`: 텍스트에 등장하는 **모든 고유 문자**(중복 제거)의 집합.
  알파벳, 공백, 쉼표, 줄바꿈 등 전부 포함됩니다.
- `sorted(...)`: 매번 실행할 때 같은 순서가 나오도록 정렬 (set은 순서가
  보장 안 되므로, 정렬을 안 하면 `stoi`의 인덱스가 실행마다 달라질 수
  있습니다).
- `stoi`/`itos`: 문자 ↔ 정수 변환 딕셔너리. 이게 우리 모델의 **character-level
  tokenizer**입니다. (단어 단위가 아니라 글자 하나하나가 토큰 1개)
- `vocab_size`: 보통 70~90 사이 (대문자/소문자/숫자/구두점/공백 등)
- `torch.tensor([stoi[ch] for ch in text], ...)`: 텍스트 전체를 정수
  시퀀스로 변환. 신경망은 문자를 직접 다루지 못하므로 필요한 변환입니다.

```python
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

- 데이터의 앞 90%를 `train_data`, 뒤 10%를 `val_data`로 나눕니다. (기법 ⑥의 일부)
- **왜 필요한가**: train loss만 보면 "모델이 데이터를 외운 것"과 "진짜
  일반화한 것"을 구분할 수 없습니다. val 데이터는 학습 중 가중치 업데이트에
  전혀 사용하지 않고, "한 번도 안 본 텍스트에서도 잘 예측하는지"를 재는
  용도로만 씁니다.

### `NextTokenDataset`

```python
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
```

- notebook 4~6과 완전히 동일한 구조입니다.
- `x`는 `idx`부터 `idx+block_size`까지의 토큰, `y`는 그걸 **한 칸 뒤로
  민** 같은 길이의 토큰입니다. 예: `block_size=4`, 텍스트가
  `"hello world"`일 때 `idx=0`이면 `x="hell"`, `y="ello"`.
- 즉 모델은 한 번에 `block_size`개의 "다음 글자 예측" 문제를 동시에
  풉니다 — `x`의 각 위치에서 그 다음 글자(`y`의 같은 위치)를 맞히는 것이
  목표입니다.
- `__len__`에서 `max(0, ...)`: 데이터 길이가 `block_size`보다 짧은 극단적
  경우에도 음수 길이가 나오지 않도록 방어.

```python
block_size = 256

train_dataset = NextTokenDataset(train_data, block_size)
val_dataset = NextTokenDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

- `block_size=256`: 모델이 한 번에 참조할 수 있는 문맥(context window)
  길이. notebook 6(64)보다 4배 넓습니다 — 더 먼 과거 문맥을 볼 수 있는
  대신, attention 연산량이 `block_size`의 제곱(O(T²))으로 늘어나서
  계산량도 늘어납니다.
- `train_loader`는 `shuffle=True`(매 epoch마다 데이터 순서를 무작위로
  섞음 — 학습에 도움), `val_loader`는 `shuffle=False`(평가는 순서가
  중요하지 않으므로 안 섞어도 무방, 일관성 있게 같은 순서로 평가).

---

## 4. Model

### 4-1. `Head` — single-head masked self-attention

```python
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
```

한 줄씩:

- `self.key/query/value = nn.Linear(emb_dim, head_size, bias=False)`:
  입력 벡터(`emb_dim`차원)를 `head_size`차원의 세 가지 다른 벡터로
  변환하는 학습 가능한 가중치 행렬 3개. **Query**는 "지금 무엇을
  찾고 있는가", **Key**는 "각 위치가 무엇을 갖고 있는가", **Value**는
  "실제로 전달할 정보"라는 비유로 설명할 수 있습니다.
- `bias=False`: attention의 Q/K/V 변환에는 보통 bias를 안 씁니다 (관행).
- `register_buffer("tril", ...)`: 학습되지 않는(=gradient가 안 흐르는)
  고정 텐서를 모델의 일부로 등록합니다. `torch.tril`은 하삼각행렬
  (좌하단만 1, 우상단은 0)을 만듭니다 — causal mask의 재료.
- `forward`에서 `B, T, C = x.shape`: 배치 크기(B), 시퀀스 길이(T), 채널
  (emb_dim, C) 차원을 각각 꺼냅니다.
- `wei = q @ k.transpose(-2, -1)`: `q`는 `(B,T,head_size)`, `k`를
  transpose하면 `(B,head_size,T)` → 행렬곱 결과는 `(B,T,T)`. 이게 바로
  "위치 i가 위치 j를 얼마나 참고해야 하는가"의 점수표(내적 = 벡터
  유사도)입니다.
- `* (k.size(-1) ** -0.5)`: head 차원 수의 제곱근으로 나눠줍니다
  (scaled dot-product attention, Attention Is All You Need 논문). 안
  나누면 차원이 클수록 내적값이 커져서 softmax가 거의 one-hot처럼
  극단적으로 변해 학습이 불안정해집니다.
- `wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))`: **causal
  mask**. `tril[:T,:T]`에서 0인 위치(미래 위치, j > i)를 `-inf`로
  덮어씁니다. softmax를 거치면 `exp(-inf) = 0`이 되어 미래 위치를 절대
  참고하지 못하게 됩니다. **이게 없으면 모델이 "정답(다음 글자)을
  미리 보고 베끼는" 것이 되어버립니다.**
- `F.softmax(wei, dim=-1)`: 각 위치(행)별로 합이 1이 되는 확률분포로
  변환.
- `self.dropout(wei)`: attention 가중치 자체에도 dropout을 적용해서
  과적합을 줄입니다.
- `out = wei @ v`: `(B,T,T) @ (B,T,head_size)` → `(B,T,head_size)`.
  attention 가중치로 value들을 가중합산한 것이 최종 출력입니다.

### 4-2. `MultiHeadAttention`

```python
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
```

- `head_size = emb_dim // num_heads`: 전체 차원을 head 개수로 나눕니다.
  예: `emb_dim=256`, `num_heads=8` → `head_size=32`. 즉 head를 여러 개
  써도 전체 연산량은 head 하나(차원 256)일 때와 비슷합니다 — 차원을
  "나눠서" 여러 관점으로 보는 것.
- `nn.ModuleList([...])`: `Head`를 `num_heads`개 만들어서 리스트로 보관.
  (그냥 Python list가 아니라 `nn.ModuleList`를 쓰는 이유: PyTorch가
  내부 파라미터들을 자동으로 추적/이동(`.to(device)`)할 수 있게 하려고.)
- `torch.cat([h(x) for h in self.heads], dim=-1)`: 각 head의 출력
  `(B,T,head_size)`을 마지막 차원 기준으로 이어붙여서 `(B,T,emb_dim)`을
  복원합니다 (head_size × num_heads = emb_dim).
- `self.proj`: 이어붙인 결과를 한 번 더 선형변환해서 head들 사이의
  정보를 섞어줍니다.
- 한 head만 쓰면 "한 가지 관점"의 관계만 학습하는데, 여러 head를 두면
  각 head가 서로 다른 패턴(예: 인접 글자 관계, 구두점 위치, 긴 문맥
  의존성 등)을 독립적으로 학습할 여지가 생깁니다.

### 4-3. `FeedForward` — 기법 ②: GELU

```python
class FeedForward(nn.Module):
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
```

- attention은 "위치들 사이의 관계"를 계산하는 역할, FeedForward는 각
  위치마다 **독립적으로** 그 정보를 가공하는 역할을 합니다 (위치를 섞지
  않고, 같은 가중치를 모든 위치에 동일하게 적용).
- `emb_dim → 4*emb_dim → emb_dim`: 차원을 4배로 확장했다가 다시 줄이는
  구조 (Transformer 논문 그대로의 비율).
- `nn.GELU()` — **여기가 notebook 6과 다른 부분**: notebook 6은
  `nn.ReLU()`를 썼습니다. ReLU는 0 미만을 칼같이 잘라내는 반면(`max(0,x)`),
  GELU는 0 근처를 부드럽게(smooth) 깎는 함수입니다. 실제 GPT-2가 GELU를
  쓰기 때문에 그대로 맞춘 것 — "GPT-2를 충실히 재현"한다는 의미가
  성능 차이보다 더 큽니다.

### 4-4. `Block` — residual + pre-norm

```python
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
```

- `x + self.sa(self.ln1(x))`: **residual connection**. attention의
  출력을 입력에 "더해줍니다"(대체하는 게 아니라). layer가 깊어질수록
  gradient가 작아지거나 불안정해지는 문제(vanishing gradient)를 줄여
  주고, 최악의 경우에도 입력이 그대로 다음 layer로 전달될 수 있는
  안전한 경로를 만들어 줍니다.
- `self.ln1(x)`를 attention/FFN에 **들어가기 전에** 적용 — 이런 방식을
  "pre-norm"이라고 부릅니다 (입력을 정규화한 뒤 attention/FFN에
  넣고, 그 결과를 정규화 전의 원본 `x`에 더함). 학습 안정성이 더
  좋다고 알려져 있고, 실제 GPT-2도 이 방식을 씁니다.
- `LayerNorm`: 각 위치의 벡터를 평균 0, 분산 1로 정규화. 학습 중 각
  layer로 들어가는 값의 스케일을 안정시킵니다.

### 4-5. `GPT2Mini.__init__`

```python
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

        self.lm_head.weight = self.token_embedding.weight   # ← 기법 ①

        self.apply(self._init_weights)
```

- `token_embedding`: 정수 인덱스(토큰) → `emb_dim`차원 벡터로 변환하는
  학습 가능한 lookup table. shape `(vocab_size, emb_dim)`.
- `position_embedding`: "이 토큰이 시퀀스에서 몇 번째 위치인가"를
  나타내는 별도의 학습 가능한 벡터. shape `(block_size, emb_dim)`.
  **왜 필요한가**: attention 연산 자체는 순서를 모릅니다 (입력 집합의
  순서를 바꿔도 attention 점수 계산 방식은 똑같습니다). 그래서 위치마다
  다른 벡터를 더해줘서 "몇 번째 글자인지" 정보를 모델에 주입합니다.
- `self.blocks = nn.Sequential(*[Block(...) for _ in range(num_layers)])`:
  `Block`을 `num_layers`(=6)개 순서대로 쌓습니다. `nn.Sequential`은
  내부적으로 각 모듈을 순서대로 통과시켜줍니다.
- `self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)`: 마지막에
  `emb_dim`차원 벡터를 `vocab_size`개의 점수(logit)로 변환 — "다음
  글자가 무엇일지"에 대한 점수표.

**기법 ① Weight tying**:
```python
self.lm_head.weight = self.token_embedding.weight
```
- `token_embedding.weight`의 shape는 `(vocab_size, emb_dim)`.
  `nn.Linear(emb_dim, vocab_size, bias=False)`의 `weight`도 PyTorch
  관행상 `(out_features, in_features) = (vocab_size, emb_dim)`로 **모양이
  정확히 같습니다.** 그래서 둘을 같은 텐서로 공유시킬 수 있습니다.
- 효과: ⓐ 파라미터 수가 `vocab_size × emb_dim`만큼 줄어듦 (이 모델
  기준 약 256 × 80 ≈ 2만 개, 전체의 일부지만 공짜로 절약). ⓑ "임베딩
  공간에서 가까운 토큰은 출력 확률에서도 비슷한 점수를 받는다"는
  자연스러운 제약이 생김 (입력 표현과 출력 표현이 같은 공간을
  공유하니까). 실제 GPT-2 논문에서도 사용하는 기법입니다.

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```
- `self.apply(self._init_weights)`: 모델 안의 **모든** 하위 모듈에 이
  함수를 재귀적으로 적용합니다.
- `Linear`/`Embedding` 가중치를 평균 0, 표준편차 0.02인 정규분포로
  초기화 — 이 숫자(0.02)는 실제 GPT-2 논문에서 그대로 가져온
  초기화 값입니다. 너무 크게 초기화하면 학습 초반 logit이 너무 커져서
  softmax가 거의 one-hot처럼 굳어버리고, 너무 작으면 신호가 약해서
  학습이 느려집니다.
- bias는 0으로 초기화 (특별한 사전 정보가 없으니 중립값에서 시작).

### 4-6. `forward`

```python
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
```

- `torch.arange(T, device=x.device)`: `[0, 1, 2, ..., T-1]` — 위치
  인덱스. `device=x.device`로 입력과 같은 장치(CPU/GPU)에 만들어야
  연산이 가능합니다.
- `tok = self.token_embedding(x)`: shape `(B, T) → (B, T, emb_dim)`.
- `pos_emb = self.position_embedding(pos)[None]`: shape `(T, emb_dim)`에
  `[None]`(=`unsqueeze(0)`)을 붙여서 `(1, T, emb_dim)`으로 만듦. 배치
  차원이 1이면 브로드캐스팅으로 모든 배치에 같은 위치 임베딩이
  더해집니다.
- `h = self.drop(tok + pos_emb)`: 토큰 임베딩과 위치 임베딩을 더한 뒤
  dropout. 이게 Transformer block들에 들어갈 최초 입력입니다.
- `h = self.blocks(h)`: 6개 `Block`을 순서대로 통과.
- `h = self.ln_f(h)`: 마지막 출력에 한 번 더 LayerNorm (GPT-2 구조에서
  마지막에 추가로 정규화를 하는 부분).
- `logits = self.lm_head(h)`: shape `(B, T, vocab_size)` — 매 위치마다
  "다음 글자"에 대한 점수 분포.
- `F.cross_entropy(logits.transpose(1, 2), targets)`: `F.cross_entropy`는
  `(B, num_classes, T)` 형태를 기대하는데 `logits`는 `(B, T,
  vocab_size)`라서 `.transpose(1, 2)`로 차원 순서를 바꿔줍니다
  (`(B, vocab_size, T)`). `targets`는 `(B, T)` shape의 정답 인덱스.
- `targets=None`이면 `loss`를 계산하지 않고 `None`을 반환 — 학습이
  아니라 순수 생성(`generate`)에만 쓸 때는 정답이 없으니까요.

### 4-7. `configure_optimizer` — 기법 ④: Decoupled weight decay

```python
def configure_optimizer(self, lr, weight_decay):
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
```

- **Weight decay**란 가중치 값 자체가 너무 커지지 않게 살짝 0 쪽으로
  당기는 정규화 기법입니다 (L2 정규화와 비슷한 효과).
- `self.named_parameters()`: 모델의 모든 파라미터를 `(이름, 텐서)`
  쌍으로 순회합니다.
- `p.dim() >= 2`: 텐서의 차원 수. `nn.Linear`/`nn.Embedding`의
  `weight`는 2차원 행렬(`(out, in)` 또는 `(vocab, emb_dim)`)이고,
  `bias`나 `LayerNorm`의 `weight`/`bias`는 1차원 벡터입니다.
- **왜 나누는가**: weight decay를 bias나 LayerNorm의 1차원 파라미터에도
  걸면, 모델의 표현력을 불필요하게 깎습니다 (이 파라미터들은 "크기"
  자체에 의미가 있는 경우가 많아서). 그래서 **2차원 이상인 행렬에만**
  decay를 걸고, 나머지는 `weight_decay=0.0`으로 둡니다. 이건
  Andrej Karpathy의 nanoGPT가 쓰는 것과 동일한 관행입니다.
- `torch.optim.AdamW(groups, ...)`: 옵티마이저에 파라미터 그룹을 2개로
  나눠서 전달 — 그룹마다 다른 `weight_decay`를 적용할 수 있는 PyTorch
  옵티마이저의 기능을 활용한 것.
- `betas=(0.9, 0.95)`: Adam의 모멘텀 계수. 기본값은 `(0.9, 0.999)`인데,
  GPT-2/GPT-3 계열은 `(0.9, 0.95)`를 즐겨 씁니다(더 최근 gradient에
  비중을 둠).

### 4-8. `generate` — 기법 ⑥: temperature + top-k

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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
```

- `@torch.no_grad()`: 생성 단계는 학습이 아니므로 gradient를 계산할
  필요가 없습니다 — 메모리/속도 절약.
- `self.eval()` / `self.train()`: `Dropout`을 끄고(`eval`) 시작해서
  생성이 끝나면 다시 켭니다(`train`). 평가/생성 시에는 dropout이
  적용되면 안 되기 때문입니다 (매번 다른 뉴런을 무작위로 끄면 생성
  결과의 일관성이 떨어짐).
- `idx_cond = idx[:, -self.block_size :]`: 지금까지 생성된 전체
  시퀀스가 `block_size`보다 길어지면, 모델이 한 번에 볼 수 있는
  최근 `block_size`개만 잘라서 씁니다 (모델 구조상 그보다 긴 문맥은
  처리할 수 없음).
- `logits = logits[:, -1, :]`: 시퀀스의 **마지막 위치**의 예측만
  꺼냅니다 — "지금까지의 글자들을 보고, 다음 한 글자는 뭘까".
- `/ temperature`: **기법 ⑥-1**. 1보다 작은 값(예: 0.8)으로 나누면
  logit들의 차이가 더 벌어져서 softmax 분포가 더 뾰족해지고(확신
  있는 토큰 위주, 더 "안전하고 일관된" 텍스트), 1보다 큰 값으로
  나누면 분포가 평평해져서 더 다양하지만 산만한 텍스트가 나옵니다.
- `top_k` 필터링: **기법 ⑥-2**. `torch.topk(logits, k)`로 확률이 가장
  높은 k개 값/인덱스를 구하고, `v[:, [-1]]`(k번째로 높은 값)보다 작은
  모든 logit을 `-inf`로 만들어 후보에서 제외합니다. → 너무 이상한
  (낮은 확률) 토큰이 뽑히는 것을 원천적으로 차단합니다.
- `torch.multinomial(probs, num_samples=1)`: 확률 분포에서 **무작위로**
  하나를 뽑습니다 (가장 확률 높은 것을 항상 고르는 게 아님 — 그래서
  실행할 때마다 결과가 조금씩 다릅니다).
- `idx = torch.cat([idx, idx_next], dim=1)`: 새로 뽑은 토큰을 시퀀스
  맨 뒤에 이어붙이고, 다음 반복에서 다시 마지막 `block_size`만큼만
  잘라 사용합니다 — 이게 **autoregressive generation**(한 글자씩
  순차적으로 생성)의 핵심 루프입니다.

---

## 5. Learning rate: warmup + cosine decay — 기법 ③

```python
def get_lr(it, warmup_iters, lr_decay_iters, min_lr, max_lr):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

세 구간으로 나뉩니다.

1. **Warmup** (`it < warmup_iters`): `max_lr * (it+1)/warmup_iters` —
   `it=0`일 때 `max_lr/warmup_iters`(거의 0)부터 시작해서, `it =
   warmup_iters - 1`일 때 거의 `max_lr`까지 **선형으로** 증가합니다.
   **왜 필요한가**: 학습 초반에는 모델 파라미터가 랜덤이라, 처음부터
   큰 lr을 쓰면 gradient가 불안정해서 loss가 튀거나 발산할 위험이
   있습니다. 작은 lr로 시작해서 모델이 어느 정도 "자리를 잡은" 뒤에
   본격적인 lr을 쓰자는 아이디어입니다.
2. **Cosine decay** (`warmup_iters <= it <= lr_decay_iters`):
   `decay_ratio`는 0(warmup 끝)에서 1(`lr_decay_iters`)까지 선형으로
   증가합니다. `math.cos(math.pi * decay_ratio)`는 `decay_ratio=0`일 때
   `cos(0)=1`, `decay_ratio=1`일 때 `cos(π)=-1`이므로, `coeff =
   0.5*(1+cos(...))`는 1에서 0으로 **부드럽게(코사인 곡선을 따라)**
   감소합니다. 최종적으로 `min_lr + coeff*(max_lr-min_lr)`는
   `max_lr`에서 `min_lr`로 매끄럽게 줄어듭니다.
3. **Decay 이후** (`it > lr_decay_iters`): 그냥 `min_lr`을 고정으로
   반환 (이 스크립트에서는 `lr_decay_iters = max_iters`라서 사실상
   학습 끝까지 cosine 구간 안에 있지만, 혹시 더 길게 돌리는 경우를
   위한 안전장치).

이 함수가 반환하는 값은 실제로 매 스텝마다 옵티마이저의 `lr`에 직접
대입됩니다 (아래 학습 루프 참고) — PyTorch의 `lr_scheduler` 객체를 쓰지
않고 직접 계산해서 넣어주는 방식입니다 (nanoGPT가 쓰는 방식과 동일).

---

## 6. 학습 루프

### 6-1. `estimate_loss`

```python
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
```

- 배치 하나만 보면 loss가 들쭉날쭉(노이즈가 큼)하기 때문에, 최대
  `eval_iters`(=20)개 배치의 loss를 모아서 평균을 냅니다 — 더 안정적인
  추정치를 얻기 위함입니다.
- `model.eval()` → 평가 → `model.train()`: dropout을 끄고 평가한 뒤
  다시 학습 모드로 복귀.

### 6-2. `get_device`

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

- NVIDIA GPU(`cuda`) → Apple Silicon GPU(`mps`) → 둘 다 없으면
  `cpu` 순서로 자동 선택합니다. `getattr(...)`로 `mps` 속성이 없는
  PyTorch 버전에서도 에러 없이 동작하도록 방어적으로 작성했습니다.

### 6-3. 메인 학습 루프

```python
model = GPT2Mini(vocab_size, block_size).to(device)
optimizer = model.configure_optimizer(max_lr, weight_decay)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % eval_interval == 0 or it == max_iters - 1:
            tr = estimate_loss(model, train_loader, device)
            va = estimate_loss(model, val_loader, device)
            ...
            print(...)
except KeyboardInterrupt:
    print("학습을 중간에 멈췄습니다. 지금까지의 결과로 계속 진행합니다.")
```

- `cycle(loader)`: `DataLoader`는 한 번 다 순회하면 끝나는데
  (`for batch in loader`가 epoch 하나만큼만 돎), `yield`를 쓴
  **무한 generator**로 감싸서 끝없이 배치를 뽑아낼 수 있게 합니다.
  이 코드는 "epoch" 단위가 아니라 **"step(iteration)" 단위**로
  학습 루프를 돌리기 때문입니다 (notebook 6의 `for epoch in
  range(...)` 방식과 다른 점).
- `for it in range(max_iters)`: 총 `max_iters`번의 학습 스텝.
- `lr = get_lr(...)` → `for g in optimizer.param_groups: g["lr"] = lr`:
  매 스텝마다 위에서 만든 warmup+cosine 공식으로 lr을 계산해서,
  옵티마이저의 **모든 파라미터 그룹**(decay 그룹, no-decay 그룹 둘 다)에
  직접 대입합니다.
- `logits, loss = model(xb, yb)`: `targets=yb`를 같이 넘겨서 forward
  안에서 loss까지 한 번에 계산 (위 4-6절 참고).
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`:
  PyTorch 학습의 표준 3단계 (기울기 초기화 → 역전파 → 파라미터 업데이트).
- `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — **기법
  ⑤: gradient clipping**. `backward()` 직후, `step()` 직전에 호출합니다.
  모든 파라미터의 gradient를 모아서 전체 norm(크기)을 계산하고, 그
  norm이 1.0을 넘으면 모든 gradient에 같은 비율을 곱해서 norm이
  정확히 1.0이 되도록 줄입니다 (방향은 유지, 크기만 제한). 가끔 한
  배치에서 비정상적으로 큰 gradient가 나와서 파라미터가 한 번에 너무
  많이 움직이는 것을 막아 학습을 안정시킵니다.
- `it % eval_interval == 0`: 매 스텝마다 평가하면 느리므로,
  `eval_interval`(=200) 스텝마다 한 번씩만 train/val loss를 측정하고
  출력합니다.
- `try / except KeyboardInterrupt`: 터미널에서 `Ctrl+C`를 누르면
  파이썬은 기본적으로 프로그램을 즉시 종료시키는데, 이렇게 감싸두면
  중간에 멈춰도 예외를 "잡아서" 그 아래 코드(loss curve 저장, 샘플
  생성)까지 정상적으로 이어서 실행됩니다.

---

## 7. Loss curve 저장

```python
plt.plot(logged_steps, train_losses, label="train loss")
plt.plot(logged_steps, val_losses, label="val loss")
...
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
try:
    plt.show()
except Exception:
    pass
```

- `logged_steps`/`train_losses`/`val_losses`는 학습 루프에서
  `eval_interval`마다 한 번씩 기록해둔 리스트입니다.
- `plt.savefig(...)`: 그래프를 항상 PNG 파일로 저장 — 어떤 실행
  환경에서든(터미널, VSCode, Colab) 결과가 파일로 남습니다.
- `plt.show()`을 `try/except`로 감싼 이유: 화면(디스플레이)이 없는
  환경(예: 일부 서버, 헤드리스 환경)에서는 `show()`가 에러를 낼 수도
  있어서, 에러가 나도 프로그램이 멈추지 않게 방어한 것입니다.

---

## 8. Sampling

```python
def generate_from_prompt(model, prompt, stoi, itos, device, max_new_tokens=500,
                          temperature=0.8, top_k=40):
    ids = [stoi[ch] for ch in prompt if ch in stoi]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    return "".join(itos[i] for i in out_ids)
```

- `[stoi[ch] for ch in prompt if ch in stoi]`: prompt 문자열을 토큰
  인덱스 리스트로 변환합니다. `if ch in stoi`로, 혹시 학습 데이터에
  없던 문자(예: 모델이 모르는 특수문자)가 prompt에 있으면 조용히
  건너뜁니다 (`KeyError`로 죽지 않게 방어).
- `torch.tensor([ids], ...)`: 배치 차원을 추가해서 `(1, len(ids))`
  shape로 만듭니다 — 모델은 항상 배치 차원을 기대하기 때문.
- `model.generate(...)`를 호출해서 새 토큰들을 이어 붙인 전체 시퀀스를
  받고, `[0]`으로 배치의 첫 번째(유일한) 시퀀스를 꺼낸 뒤
  `.tolist()`로 Python list로 변환합니다.
- `"".join(itos[i] for i in out_ids)`: 토큰 인덱스 리스트를 다시 문자
  하나하나로 변환해서 이어붙입니다 — 사람이 읽을 수 있는 텍스트로 복원.

```python
torch.save(model.state_dict(), "gpt2_mini_frankenstein.pt")
```
- `model.state_dict()`: 모델의 모든 학습된 가중치를 담은 딕셔너리.
  `torch.save`로 디스크에 저장해두면, 나중에 `model.load_state_dict(
  torch.load(...))`로 다시 불러와서 재학습 없이 바로 생성에 쓸 수
  있습니다.

---

## 핵심 하이퍼파라미터 한눈에 보기

| 이름 | 값 | 의미 |
|---|---|---|
| `block_size` | 256 | 한 번에 보는 문맥 길이 (토큰 개수) |
| `emb_dim` | 256 | 토큰 하나를 표현하는 벡터의 차원 |
| `num_heads` | 8 | attention head 개수 (head_size = 256/8 = 32) |
| `num_layers` | 6 | Transformer block을 쌓는 개수 |
| `dropout` | 0.2 | 학습 중 무작위로 끄는 뉴런의 비율 |
| `batch_size` | 64 | 한 스텝에 동시에 처리하는 시퀀스 개수 |
| `max_lr` / `min_lr` | 3e-4 / 3e-5 | learning rate의 최대/최소값 |
| `warmup_iters` | 100 | lr을 선형으로 끌어올리는 스텝 수 |
| `max_iters` | 3000 | 총 학습 스텝 수 |
| `weight_decay` | 0.1 | 2차원 이상 가중치에 적용하는 decay 강도 |
| grad clip norm | 1.0 | gradient의 최대 허용 norm |

---

## Q&A

**Q1. 이 모델이 notebook 6과 가장 크게 다른 점은?**
구조(attention/residual/layernorm)는 동일하고, "학습 레시피"가
다릅니다 — weight tying, GELU, lr 스케줄, decoupled weight decay,
gradient clipping, train/val split + temperature/top-k sampling 6가지를
추가했습니다. 데이터셋도 Tiny Shakespeare에서 Frankenstein으로
바꿨습니다.

**Q2. weight tying이 왜 가능한가요? 임베딩과 출력층은 역할이 다른데?**
두 행렬의 shape가 우연이 아니라 구조적으로 `(vocab_size, emb_dim)`로
**항상 같습니다.** 입력 임베딩은 "토큰 → 벡터", 출력층은 "벡터 →
토큰별 점수"로 방향이 반대지만, 같은 행렬을 두 방향으로(한쪽은 그대로,
한쪽은 전치 없이 Linear의 관행대로) 재사용할 수 있습니다.

**Q3. Warmup이 없으면 어떻게 되나요?**
초반에 큰 lr로 무작위 초기화된 가중치를 갑자기 크게 업데이트하면
loss가 발산하거나 매우 불안정해질 수 있습니다. 작은 모델/적은
데이터에서는 영향이 작을 수도 있지만, 안전장치로 넣었습니다.

**Q4. Gradient clipping과 weight decay는 둘 다 "값을 작게 누르는"
건데 뭐가 다른가요?**
Weight decay는 **파라미터(가중치) 값** 자체를 학습 내내 천천히 0
쪽으로 당기는 정규화이고, gradient clipping은 **한 스텝의
업데이트 크기(gradient)** 가 너무 커지는 것을 그 스텝에서만 순간적으로
제한하는 것입니다. 목적과 작동 시점이 다릅니다.

**Q5. top-k와 temperature를 같이 쓰면 순서가 중요한가요?**
이 코드에서는 `logits`를 `temperature`로 나눈 **다음** top-k 필터링을
합니다. temperature는 분포의 "뾰족함"을 조절하고, top-k는 후보 자체를
줄이는 것이라 순서를 바꿔도 최종 결과(어떤 토큰이 후보에 남는지)는
같지만, 일관성을 위해 이 순서로 고정했습니다.

**Q6. `cycle()` 함수는 왜 필요한가요? 그냥 epoch로 돌리면 안 되나요?**
가능합니다 (notebook 6은 실제로 epoch 단위입니다). 이 코드는 "warmup
스텝 수", "총 스텝 수" 같은 개념을 epoch 경계와 무관하게 다루기
편하게 하려고 step 단위 루프로 작성했습니다. `cycle()`은 `DataLoader`를
무한 반복시켜서 epoch 경계를 신경 쓰지 않고 원하는 스텝 수만큼 학습할
수 있게 해주는 헬퍼입니다.

**Q7. `block_size=256`인데 `max_iters=3000`이면 데이터를 몇 번이나
보는 건가요?**
`batch_size=64 × block_size=256 × max_iters=3000`만큼의 토큰을
처리합니다(중복 포함). 실제 train 토큰 수와 비교하면 데이터를 몇
바퀴 도는지 추정할 수 있습니다 (train 토큰 수는 실행 시 출력되는
`"train tokens: ..."` 값을 확인하세요).

**Q8. 이 모델로 생성한 텍스트가 완벽한 영어 문장이 아닌 이유는?**
character 단위 토큰화 + 약 5~6M 파라미터 규모의 작은 모델 + 소설
한 권(약 40만 글자) 분량의 데이터로는, 실제 GPT-2(파라미터 1억~15억,
데이터 수십 GB)만큼의 문장 완성도를 기대할 수 없습니다. "영어
단어처럼 보이는 패턴과 소설 특유의 어휘"가 나오는 정도가 이 스케일에서
합리적으로 기대할 수 있는 수준입니다.
