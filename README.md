# Tiny GPT: Bigram → MLP → Self-Attention → GPT 2.0

문자 단위(character-level) 언어모델을 가장 단순한 형태(bigram)부터 시작해서
GPT 구조까지 한 단계씩 쌓아 올리는 프로젝트입니다. 수업에서 다룬
notebook 1~6의 흐름을 그대로 보존하고, 마지막에 **GPT 2.0**(notebook 7)을
추가했습니다.

**GPT 2.0의 정의: notebook 6의 `TinyGPT` 구조를 그대로 가져와서, Tiny
Shakespeare가 아닌 다른 데이터셋(Frankenstein)으로 학습시키고, 그 결과를
train/val loss 그래프와 생성 샘플로 보여주는 것.** 구조 자체는 일부러
바꾸지 않았습니다.

## 프로젝트 구조

```
.
├── notebook_01.py        # Bigram language model (names.txt)
├── notebook_02.py        # MLP + embedding (names.txt)
├── notebook_03.py        # 같은 MLP를 Tiny Shakespeare에 적용
├── notebook_04.py        # GPT-style dataset (target도 sequence)
├── notebook_05.py        # Single-head masked self-attention
├── notebook_06.py        # Multi-head attention + FFN + residual + LN = TinyGPT
├── notebook_07_gpt2.py   # ★ GPT 2.0: Frankenstein 데이터셋 + train/val split
├── requirements.txt      # pip install -r requirements.txt
├── .gitignore            # 다운로드 데이터/체크포인트는 git에 안 올림
├── loss_curve.png        # (notebook_07 실행 후 생성되는 학습 곡선)
└── README.md
```

## 전체 흐름 한눈에 보기

| # | 핵심 질문 | 입력 → 출력 | 새로 추가된 것 |
|---|---|---|---|
| 1 | 가장 단순한 LM은? | 문자 1개 → 다음 문자 1개 | one-hot, bigram 행렬 |
| 2 | context를 늘리면? | 문자 N개 → 다음 문자 1개 | embedding, MLP |
| 3 | 데이터를 키우면? | 위와 동일 | Tiny Shakespeare |
| 4 | target도 시퀀스라면? | 시퀀스 → 시퀀스 | positional embedding, `(B,T,V)` 출력 |
| 5 | 위치마다 다른 곳을 보게 하려면? | 시퀀스 → 시퀀스 | Q/K/V, causal mask, self-attention |
| 6 | 여러 관점에서 보게 하려면? | 시퀀스 → 시퀀스 | multi-head, FFN, residual, LayerNorm |
| 7 | **GPT 2.0** | 시퀀스 → 시퀀스 | **다른 데이터셋 + train/val 평가** |

notebook 1~6은 "**구조**를 단계적으로 완성해가는 과정"이고, notebook 7(GPT 2.0)은
"완성된 구조를 **다른 데이터로 학습시키고 결과를 검증**하는 과정"입니다.

## GPT 2.0에서 무엇이 바뀌었나

| 항목 | notebook 6 (TinyGPT) | notebook 7 (GPT 2.0) |
|---|---|---|
| 모델 구조 (`Head`/`MultiHeadAttention`/`FeedForward`/`Block`/`TinyGPT`) | - | **동일** |
| 데이터셋 | Tiny Shakespeare (극본체) | **Frankenstein**, Mary Shelley 1818 (1인칭 소설체) |
| 평가 방식 | train loss만 출력 | **train/val 90:10 split**, 둘 다 추적해서 그래프로 제시 |

### 왜 구조를 안 바꿨나

처음에는 weight tying, GELU, learning rate warmup/cosine decay, decoupled
weight decay 같은 실제 GPT-2의 학습 기법들을 추가하는 버전을 만들었는데,
**과제 범위에 비해 변경 사항이 너무 많아서** 다시 정리했습니다.

그래서 GPT 2.0은 **"다른 데이터셋으로 같은 구조를 학습시키고, 그 결과를
제대로 검증한다"**는 원래 과제 요구사항에만 집중했습니다. 모델 구조
(`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `TinyGPT`)는
notebook 6과 다르지 않습니다.

## 데이터셋 출처 및 라이선스

- **Frankenstein; or, The Modern Prometheus** (Mary Shelley, 1818)
- 출처: [Project Gutenberg eBook #84](https://www.gutenberg.org/ebooks/84)
- 미국 저작권 만료로 **퍼블릭 도메인**입니다. 누구나 자유롭게 사용/배포할 수 있습니다.
- 다운로드: `https://www.gutenberg.org/cache/epub/84/pg84.txt`
  (스크립트가 자동으로 다운로드하고, 앞뒤 Gutenberg 안내문은 자동으로 제거합니다.
  이 주소가 막히면 `https://www.gutenberg.org/files/84/84-0.txt` 미러로 자동 재시도합니다.)

---

## 코드 상세 설명

`notebook_07_gpt2.py`를 위에서부터 순서대로 설명합니다. 각 섹션은 파일의
`# %%` 셀과 그대로 대응됩니다.

### 0. 데이터 다운로드 & 정제

```python
download_text(DATA_PATH, GUTENBERG_URLS)
raw_text = DATA_PATH.read_text(encoding="utf-8")
text = strip_gutenberg_boilerplate(raw_text)
```

- `download_text`: Project Gutenberg에서 텍스트 파일을 받습니다. 이미
  `frankenstein.txt`가 있으면 다시 받지 않습니다(`if path.exists(): return`).
  URL을 두 개 등록해놓고, 첫 번째가 실패하면 두 번째를 시도합니다.
- `strip_gutenberg_boilerplate`: Gutenberg가 배포하는 텍스트 파일은 앞에
  "이 책은 퍼블릭 도메인입니다..." 같은 법적 안내문이 붙고, 뒤에도
  비슷한 안내문이 붙습니다. 정규식으로 `*** START OF THE PROJECT
  GUTENBERG EBOOK ***`와 `*** END OF ... ***` 사이만 잘라냅니다. 이걸
  안 하면 모델이 소설 문체가 아니라 라이선스 안내문 문체까지 학습하게
  됩니다.
- notebook 1~6은 `input.txt`/`names.txt`가 이미 정제된 채로 제공돼서
  이 단계가 없었습니다. **다른 데이터셋을 직접 가져올 때 필요한 추가
  단계**라고 설명하면 됩니다.

### 1. Vocab + Dataset

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
...
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

- `chars`: 텍스트에 등장하는 모든 "문자"(알파벳, 공백, 구두점 등)의
  집합. 이게 우리 모델의 vocabulary입니다. (단어 단위가 아니라 **문자
  단위** tokenizer라서, 'a'도 토큰 1개, 'z'도 토큰 1개입니다.)
- `stoi`/`itos`: 문자 ↔ 정수 인덱스 변환 딕셔너리. 신경망은 문자를 직접
  다루지 못하고 숫자(정수 인덱스 → 나중에 embedding 벡터)만 다룰 수
  있어서 필요합니다.
- **train/val split (90:10)**: notebook 6에는 없던 부분. 데이터의
  앞 90%로 학습하고, 뒤 10%는 학습에 전혀 쓰지 않고 "모델이 한 번도
  못 본 텍스트에서도 잘 예측하는지" 확인하는 용도로만 씁니다. 이게
  없으면 train loss가 낮아도 그게 "이해해서 낮은 건지, 그냥 외워서
  낮은 건지" 구분할 수 없습니다.
- `NextTokenDataset`: 길이 `block_size`인 문자열 `x`와, 그걸 한 칸씩 밀어낸
  `y`를 만듭니다. 예를 들어 `block_size=4`일 때 `"hello"`에서
  `x="hell"`, `y="ello"`. 즉 모델은 매 위치에서 "다음 문자가 뭘까"를
  동시에 여러 개 예측하도록 학습됩니다 (notebook 4에서 처음 등장한 구조).

### 2. Model

`Head` → `MultiHeadAttention` → `FeedForward` → `Block` → `TinyGPT`,
notebook 6과 글자 단위로 동일합니다. 각각의 역할:

**`Head`** — single-head self-attention.
```python
k = self.key(x); q = self.query(x); v = self.value(x)
wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ v
```
- `key`, `query`, `value`: 같은 입력 `x`를 세 개의 다른 선형변환에
  통과시켜 만든 벡터. **Query**는 "내가 지금 무엇을 찾고 있는가",
  **Key**는 "나는 무엇을 갖고 있는가", **Value**는 "내가 실제로
  전달할 정보"라고 비유적으로 설명할 수 있습니다.
- `q @ k.transpose(-2, -1)`: 모든 위치 쌍 (i, j)에 대해 "위치 i가
  위치 j를 얼마나 참고해야 하는가"를 점수로 계산합니다 (내적 = 유사도).
- `* (k.size(-1) ** -0.5)`: head 차원 수의 제곱근으로 나눠서 softmax
  입력값이 너무 커지지 않게 합니다 (Attention is All You Need 논문의
  scaled dot-product attention).
- `masked_fill(... -inf)` — **causal mask**: 위치 i는 자기 자신과 그
  이전 위치(j ≤ i)만 봐야 합니다. 미래 위치는 `-inf`로 채워서 softmax
  후 확률이 0이 되게 만듭니다. 이게 없으면 모델이 "정답(다음 글자)을
  미리 보고 베끼는" 것이 되어버립니다.
- `wei @ v`: 계산된 가중치로 value들을 가중합산 — 이게 attention의
  최종 출력입니다.

**`MultiHeadAttention`** — `Head`를 여러 개(`num_heads`) 병렬로 돌리고
결과를 이어붙인(`torch.cat`) 다음 선형변환(`proj`)으로 다시 합칩니다.
한 head만 쓰면 "한 가지 관점"으로만 문맥을 보는데, 여러 head를 쓰면
각 head가 서로 다른 패턴(예: 문법적 관계, 단어 거리 등)을 따로 학습할
여지가 생깁니다.

**`FeedForward`** — 두 개의 Linear와 ReLU로 구성된 단순한 MLP.
attention은 "위치들 사이의 관계"를 계산하고, FeedForward는 각 위치마다
독립적으로 "그 정보를 가공"합니다. 차원을 4배로 늘렸다가 다시 줄이는
구조(`emb_dim → 4*emb_dim → emb_dim`)는 Transformer 논문 그대로입니다.

**`Block`** — attention과 feedforward를 감싸는 residual + LayerNorm:
```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```
- `x + ...` (residual connection): 원래 입력을 그대로 더해줘서, layer가
  깊어져도 gradient가 잘 흐르게 합니다. (학습이 잘 안 되면 attention/FFN이
  "아무것도 안 하는 것"에 가깝게 수렴해도 최소한 입력은 그대로 통과하니까
  안전망 역할을 합니다.)
- `LayerNorm`: 각 위치의 벡터를 평균 0, 분산 1로 정규화해서 학습을
  안정시킵니다. attention/FFN에 들어가기 *전에* 정규화하는 방식("pre-norm")을
  씁니다.

**`TinyGPT`** — token embedding + position embedding을 더한 뒤
`Block`을 여러 개(`num_layers`) 쌓고, 마지막에 `lm_head`(Linear)로
vocab 크기만큼의 점수(logit)를 출력합니다.
- `position_embedding`이 필요한 이유: attention 연산 자체는 "위치"
  정보를 모릅니다(순서를 바꿔도 같은 집합이면 같은 결과). 그래서 각
  위치마다 별도의 학습 가능한 벡터를 더해줘서 "몇 번째 글자인지"
  정보를 모델에 알려줍니다.

### 3. 학습 루프

```python
def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    ...
    loss.backward()
    optimizer.step()
    ...

@torch.no_grad()
def evaluate(model, loader, device, max_steps=None):
    ...   # backward/step 없음
```
- `sequence_cross_entropy`: 모델 출력 `logits`의 shape는
  `(B, T, vocab_size)`인데, `F.cross_entropy`는 `(B, vocab_size, T)`
  형태를 기대해서 `transpose(1, 2)`로 차원을 바꿔줍니다.
- `train_one_epoch`과 `evaluate`는 **구조가 완전히 동일**합니다. 차이는
  딱 세 줄(`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`)과
  `@torch.no_grad()` 데코레이터입니다. "학습"과 "평가"의 코드 차이가
  본질적으로 "가중치를 업데이트하느냐 마느냐" 뿐이라는 걸 보여줍니다.
- `optimizer = torch.optim.AdamW(...)`: notebook 6과 동일하게 AdamW,
  learning rate `3e-4`. (Adam에 weight decay를 더 올바르게 적용한
  변형입니다.)
- 매 epoch마다 `train_one_epoch`(학습 진행) → `evaluate`(val set으로 채점)를
  반복하면서 두 loss를 모두 기록합니다.

### 4. Loss curve

`train_history`/`val_history`를 그래프로 그립니다. 둘 다 떨어지면
"학습이 잘 되고 있다"는 뜻이고, train loss는 계속 떨어지는데 val loss가
어느 순간부터 다시 올라가면 그게 **과적합(overfitting)** 신호입니다 —
모델이 train 데이터를 "이해"하는 대신 "외우기" 시작했다는 뜻입니다.
데이터가 Tiny Shakespeare보다 적기 때문에 이 현상이 더 빨리 나타날
수 있습니다 (실행 후 실제 그래프를 보고 이 부분을 직접 언급하면 좋습니다).

### 5. Sampling

`sample_gpt`는 notebook 6의 생성 함수와 동일합니다.
- 길이 `block_size`짜리 context를 0(패딩)으로 채운 뒤, prompt 문자를
  하나씩 넣어 context를 채웁니다.
- 매 스텝마다: 모델에 현재 context를 넣어 마지막 위치의 logits만
  꺼내고, softmax로 확률분포를 만든 다음, `torch.multinomial`로
  그 분포에서 한 글자를 무작위로 뽑습니다 (가장 확률 높은 것을 항상
  고르는 게 아니라 분포에서 샘플링 — 매번 실행할 때마다 결과가 조금씩
  다른 이유입니다).
- 새로 뽑은 글자를 context 맨 뒤에 추가하고, 맨 앞 글자는 버립니다
  (`context[:, 1:]`) — 항상 길이 `block_size`를 유지합니다.

---

## 실행 방법 (VSCode / 로컬 환경)

### 1. 가상환경 만들기

```bash
python -m venv .venv
source .venv/bin/activate      # Windows는: .venv\Scripts\activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

NVIDIA GPU가 있는 Windows/Linux라면, CUDA 버전에 맞는 PyTorch를
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)에서 확인해서
대신 설치하면 학습이 더 빨라집니다 (없어도 CPU로 잘 동작합니다).

### 3. VSCode에서 열기

```bash
code .
```

`notebook_07_gpt2.py`를 열고, 우측 하단에서 방금 만든 `.venv` 인터프리터를
선택하세요 (Python 확장이 설치되어 있어야 합니다).

### 4. 실행

이 파일은 `# %%`로 셀이 나뉘어 있어서, VSCode + Jupyter 확장을 쓰면 각 셀
위에 뜨는 **"Run Cell"** 버튼으로 한 셀씩 실행할 수 있습니다 (그래프도
바로 아래에 나타남). 또는 터미널에서 통째로:

```bash
python notebook_07_gpt2.py
```

### 5. 실행 후 생성되는 파일

- `frankenstein.txt` — 다운로드/정제된 학습 데이터 (재실행 시 재사용)
- `loss_curve.png` — train/val loss 그래프
- `tinygpt_frankenstein.pt` — 학습된 모델 체크포인트

`frankenstein.txt`와 `.pt` 체크포인트는 `.gitignore`에 들어있어서 GitHub에는
올라가지 않습니다 (스크립트를 다시 돌리면 똑같이 재생성되기 때문). `loss_curve.png`는
README에 첨부할 용도라 그대로 커밋하면 됩니다.



---

## 학습 시간이 너무 오래 걸릴 때

코드 안에 시간을 조절하는 숫자 두 개가 있습니다 (3. 학습 섹션):

```python
NUM_EPOCHS = 15            # notebook_06은 100
MAX_STEPS_PER_EPOCH = 200  # notebook_06은 300
```

- **가장 빠른 해결책**: 이 두 숫자를 줄이세요. 예: `NUM_EPOCHS = 5`,
  `MAX_STEPS_PER_EPOCH = 100` → 전체 스텝이 1/6 수준으로 줄어듭니다.
  (총 학습 스텝 ≈ `NUM_EPOCHS × MAX_STEPS_PER_EPOCH`)
- **중간에 멈춰도 괜찮습니다**: 터미널에서 `Ctrl + C`를 누르면 학습을
  바로 멈추고, 그때까지 학습된 모델로 loss curve와 생성 샘플까지
  마무리합니다 (`try/except KeyboardInterrupt`로 처리해놨습니다).
- 이 모델은 notebook 6과 똑같은 크기(`emb_dim=128`, `num_layers=4`,
  `block_size=64`, 파라미터 약 0.8M)라서 GPU 없이 CPU로 돌려도 크게
  오래 걸리지 않습니다 — 보통 노트북 기준 몇 분 내로 끝납니다.
- 그래도 느리다면 `block_size`를 32로 줄이는 것도 방법입니다 (단,
  `Head` 클래스의 causal mask 크기도 같이 줄어드는 거라 모델이 "기억할
  수 있는 문맥"이 짧아진다는 점은 설명할 수 있어야 합니다).

---

## 결과
raw length (boilerplate 포함): 438841
cleaned length (본문만): 419336
Frankenstein;

or, the Modern Prometheus

by Mary Wollstonecraft (Godwin) Shelley

 CONTENTS

 Letter 1
 Letter 2
 Letter 3
 Letter 4
 Chapter 1
 Chapter 2
 Chapter 3
 Chapter 4
 Chapter 5
 Chapter 6
 Chapter 7
 Chapter 8
 Chapter 9
 Chapter 10
 Chapter 11
 Chapter 12
 Chapter 13
 Chapter 14
 Chapt

vocab_size: 83
train tokens: 377402 | val tokens: 41934
xb.shape: torch.Size([64, 64]) | yb.shape: torch.Size([64, 64])
model parameters: 0.82M

epoch  0 | train loss 2.7201 | val loss 2.4095 |   5.9s elapsed
epoch  1 | train loss 2.3758 | val loss 2.2483 |  12.3s elapsed
epoch  2 | train loss 2.1933 | val loss 2.0513 |  17.3s elapsed
epoch  3 | train loss 2.0479 | val loss 1.9232 |  23.0s elapsed
epoch  4 | train loss 1.9483 | val loss 1.8296 |  27.9s elapsed
epoch  5 | train loss 1.8739 | val loss 1.7643 |  33.6s elapsed
epoch  6 | train loss 1.8139 | val loss 1.7133 |  38.6s elapsed
epoch  7 | train loss 1.7642 | val loss 1.6695 |  43.8s elapsed
epoch  8 | train loss 1.7248 | val loss 1.6374 |  49.2s elapsed
epoch  9 | train loss 1.6854 | val loss 1.6126 |  54.2s elapsed
epoch 10 | train loss 1.6537 | val loss 1.5887 |  59.7s elapsed
epoch 11 | train loss 1.6239 | val loss 1.5697 |  64.6s elapsed
epoch 12 | train loss 1.6040 | val loss 1.5497 |  70.1s elapsed
epoch 13 | train loss 1.5837 | val loss 1.5244 |  74.9s elapsed
epoch 14 | train loss 1.5620 | val loss 1.5123 |  81.3s elapsed

<img width="691" height="470" alt="gpt2 0그래프" src="https://github.com/user-attachments/assets/a5c9c528-2462-4290-a422-5815a61f343f" />

### 생성 샘플

```
It was on a dreary night any educes and the remained which her passed
away resently the affection of blight, talk me did were resiant of
which he realing culting poil deaving.”

My some, Chapter more deear do tank what he not pleasure resity remain since
of a feelings feling on on I befection. He
dowever to pereceived on that vill besidence, but list which from the impoorse that
attreat ewellected my heart? Saving that h
saved tinygpt_frankenstein.pt
```

---

## Q&A

**Q1. GPT 2.0에서 정확히 뭘 바꿨나요?**
딱 두 가지입니다. (1) 데이터셋을 Tiny Shakespeare에서 Frankenstein으로
바꿨고, (2) 데이터를 90:10으로 train/val로 나눠서 둘 다 loss를
추적했습니다. 모델 구조(`Head`, `MultiHeadAttention`, `FeedForward`,
`Block`, `TinyGPT`)는 notebook 6과 동일합니다.

**Q2. self-attention에서 Q, K, V는 뭔가요?**
같은 입력 벡터를 세 개의 다른 가중치 행렬에 통과시켜 만든 세 가지
버전입니다. Query는 "지금 무엇을 찾고 있는지", Key는 "각 위치가 무엇을
갖고 있는지", Value는 "실제로 전달할 정보"입니다. Query와 Key의 내적으로
유사도(=얼마나 주목할지)를 계산하고, 그 가중치로 Value를 합산합니다.

**Q3. causal mask(`tril`)는 왜 필요한가요?**
언어모델은 "다음 글자 예측"을 학습하는데, 만약 위치 i가 자기보다
미래(j > i)의 글자를 attention으로 들여다볼 수 있다면, 모델이 정답을
미리 보고 그대로 베끼는 것과 같아집니다. `tril`(하삼각행렬)로 미래
위치의 attention 점수를 `-inf`로 만들어서 softmax 후 확률이 0이
되게 합니다.

**Q4. multi-head는 왜 head 하나로 안 하나요?**
head 하나는 "한 가지 관점"의 관계만 학습합니다. head를 여러 개 두면
각 head가 서로 다른 종류의 패턴(예: 바로 앞 글자와의 관계, 문장 구조,
구두점 위치 등)을 독립적으로 학습할 수 있는 여지가 생깁니다. 전체
차원(`emb_dim`)을 head 수만큼 나눠서 쓰기 때문에 계산량은 head 하나일
때와 비슷합니다.

**Q5. train/val split을 왜 추가했나요?**
notebook 6은 train loss만 봤는데, train loss는 "모델이 학습 데이터를
얼마나 잘 외웠는지"만 알려줍니다. val(검증) 데이터는 학습에 전혀 안
쓰기 때문에, val loss가 같이 떨어지는지를 보면 모델이 실제로 일반화된
패턴을 배우는지, 그냥 외우는지 구분할 수 있습니다.

**Q6. 생성된 문장이 완벽한 영어가 아닌데 왜 그런가요?**
이 모델은 (1) 글자(character) 단위로 학습하고, (2) 파라미터가 약
0.8M밖에 안 되는 아주 작은 모델이고, (3) 학습 데이터도 Frankenstein
한 권(약 40만 글자)뿐입니다. 실제 GPT-2는 파라미터가 1억~15억 개,
학습 데이터는 수십 GB 규모입니다. 그래서 완벽한 문장보다는 "영어
단어처럼 보이는 패턴, 등장인물 이름, 소설 특유의 어휘"가 나오는 정도가
이 스케일에서 기대할 수 있는 결과입니다.

**Q7. 왜 character 단위로 토큰화했나요? 단어 단위로 하면 안 되나요?**
notebook 1부터 이어진 설계라 일관성을 유지했습니다. character 단위는
vocab 크기가 작아서(보통 60~100개) 구현이 단순하고, 모르는 단어
("OOV", out-of-vocabulary) 문제가 없습니다. 단점은 같은 의미를 표현하는
데 더 많은 토큰이 필요해서 비효율적이라는 점이고, 실제 GPT 계열
모델들은 BPE 같은 subword tokenization을 사용합니다.

**Q8. residual connection(`x + ...`)이 왜 필요한가요?**
layer를 깊게 쌓을수록 gradient가 역전파 과정에서 점점 작아지거나
불안정해지는 문제가 있습니다(vanishing gradient). 입력을 그대로 더해주는
경로(residual)가 있으면, 각 layer는 "입력에 무엇을 더할지"만 학습하면
되고, 최악의 경우에도 입력이 그대로 다음 layer로 전달될 수 있는
안전한 경로가 생깁니다.

**Q9. Frankenstein을 선택한 이유는?**
저작권이 만료된 퍼블릭 도메인 텍스트이면서, Tiny Shakespeare(대사+지문
형식의 극본체)와는 완전히 다른 1인칭 소설체라서, 같은 구조가 다른
문체에서도 동작하는지 확인하기에 적합하다고 판단했습니다.

---

## 다음에 시도해볼 수 있는 것

- 다른 데이터셋으로 교체 (한국어 텍스트, 다른 소설 등 — 저작권 있는
  텍스트는 학습용으로만 사용하고 생성물을 그대로 배포하지 않도록 주의)
- character-level → BPE/subword tokenization으로 교체
- `emb_dim`, `num_layers`, `num_heads`를 키워서 스케일링 효과 관찰
- weight tying, learning rate scheduling 등 실제 GPT-2 트릭 추가 적용

## 참고

이 프로젝트는 Andrej Karpathy의 [nanoGPT](https://github.com/karpathy/nanoGPT) /
[makemore](https://github.com/karpathy/makemore) 스타일의 교육용 구현 흐름을
따릅니다. 코드는 수업용 notebook 1~6을 기반으로 직접 확장한 것입니다.
