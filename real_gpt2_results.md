# GPT 2.0 (Full) 실행 결과 — Colab GPU 학습

## 실행 환경

- 플랫폼: Google Colab
- Device: `cuda` (GPU)
- 실행 파일: `real_gpt2.0.py`
- 데이터셋: *Frankenstein* (Mary Shelley, 1818, Project Gutenberg #84)
- 주요 하이퍼파라미터: `block_size=256`, `emb_dim=256`, `num_heads=8`,
  `num_layers=6`, `batch_size=64`, `max_iters=3000`, `warmup_iters=100`,
  `max_lr=3e-4`, `min_lr=3e-5`, `weight_decay=0.1`, gradient clip norm `1.0`

## 학습 곡선 (train / val loss)

| step | learning rate | train loss | val loss |
|---|---|---|---|
| 0 | 3.00e-06 | 4.4901 | 4.4727 |
| 200 | 2.99e-04 | 2.4020 | 2.4046 |
| 400 | 2.93e-04 | 2.2031 | 2.2164 |
| 600 | 2.81e-04 | 1.9537 | 1.9696 |
| 800 | 2.63e-04 | 1.7672 | 1.7943 |
| 1000 | 2.41e-04 | 1.6319 | 1.6807 |
| 1200 | 2.15e-04 | 1.5150 | 1.5965 |
| 1400 | 1.87e-04 | 1.4454 | 1.5305 |
| 1600 | 1.58e-04 | 1.3919 | 1.4967 |
| 1800 | 1.29e-04 | 1.3369 | 1.4580 |
| 2000 | 1.02e-04 | 1.3055 | 1.4285 |
| 2200 | 7.76e-05 | 1.2826 | 1.4173 |
| 2400 | 5.75e-05 | 1.2616 | 1.4087 |
| 2600 | 4.25e-05 | 1.2480 | 1.3967 |
| 2800 | 3.32e-05 | 1.2358 | 1.3857 |
| 2999 | 3.00e-05 | 1.2269 | 1.3844 |

> `loss_curve.png`(Colab에서 자동 생성된 그래프 파일)를 다운로드해서 이
> 문서와 같은 폴더에 넣으면, 아래 줄로 GitHub에서 그래프가 바로 보입니다.
> ```
> ![training curve](loss_curve.png)
> ```

## 최종 결과

- 최종 train loss: **1.2269**
- 최종 val loss: **1.3844**
- val perplexity: **3.99** (`exp(1.3844) ≈ 3.99`)
  → 모델이 다음 글자를 예측할 때, 평균적으로 약 4개 후보 중에서 고르는
  수준의 확신도를 갖고 있다는 뜻입니다. (참고로 아무것도 모르는 상태,
  즉 vocab 전체에서 완전히 무작위로 찍는 경우의 perplexity는
  `vocab_size`(이 모델은 약 80) 근처입니다 — 학습 전 step 0의 loss
  4.49가 `ln(80) ≈ 4.38`에 가까운 것도 이를 보여줍니다.)

## 관찰 — Learning rate 스케줄

- step 0: lr 3.00e-06 (warmup 시작, 거의 0)
- step 200경: lr 2.99e-04 (`max_lr=3e-4`에 거의 도달 — warmup 100 step
  안에 빠르게 상승)
- 이후 step이 진행될수록 lr이 cosine 곡선을 따라 부드럽게 감소
  (2.93e-04 → 2.81e-04 → ... → 3.00e-05)
- 코드에서 설계한 "warmup(100 step) → cosine decay(나머지 2900 step)"가
  로그 수치 그대로 확인됩니다.

## 관찰 — Train/Val Loss 격차 (과적합 여부)

| step | train | val | 격차(val − train) |
|---|---|---|---|
| 0 | 4.4901 | 4.4727 | -0.017 |
| 1000 | 1.6319 | 1.6807 | +0.049 |
| 2000 | 1.3055 | 1.4285 | +0.123 |
| 2999 | 1.2269 | 1.3844 | +0.158 |

- 초반(step 0~600 정도)에는 train/val loss가 거의 같이 떨어집니다 —
  아직 모델이 데이터를 "외우기"보다는 일반적인 언어 패턴(글자 빈도,
  흔한 글자 조합 등)을 배우는 단계입니다.
- step 1000을 넘기면서 격차가 서서히 커집니다 (+0.05 → +0.12 → +0.16).
  train loss는 계속 떨어지는데 val loss는 더 느리게 떨어지거나
  정체되는 — **약한 과적합** 신호입니다. Frankenstein 데이터가 Tiny
  Shakespeare보다 적기 때문에 비교적 빠르게 나타날 수 있는 자연스러운
  현상입니다.
- 다만 val loss 자체도 끝까지 꾸준히 낮아지고 있어서(1.68 → 1.43 → 1.38),
  심각한 과적합(val loss가 다시 올라가는 단계)까지는 가지 않은
  상태에서 학습이 종료되었습니다.

## 생성 샘플

**Prompt**: `"It was on a dreary night of November"`

```
It was on a dreary night of November. A father firsted by I precipiced
the body of the impression of senses while I had proceded the same
bitter and despert the suddenly broke and did some of with the hands
of the beautiful of the expression and enterprise life to the free
not of modious like by the life, I was alimned preceived my heart
life but smiles in my remains.  He fiend, I as fear hours after them,
I began the of the crimes of his fave disstant for a largue of the
months of extrements and for she had ald not succeived to m
```

**관찰**: 완전히 문법적인 영어 문장은 아니지만, 다음과 같은 점에서
character-level 모델이 소설의 문체적 패턴을 어느 정도 학습했다는 것을
확인할 수 있습니다.

- 실제 영어 단어가 다수 등장 (father, body, impression, senses, bitter,
  suddenly, broke, hands, beautiful, expression, life, heart, smiles,
  remains, fiend, fear, hours, crimes, months 등)
- Frankenstein 특유의 어휘/주제(fiend, crimes, heart, life, remains)가
  자연스럽게 섞여 나옴 — Tiny Shakespeare로 학습했다면 안 나왔을
  단어들
- 문장 부호(쉼표, 마침표)와 대문자 사용 패턴(문장 시작 시 대문자,
  "I" 항상 대문자)도 대체로 올바름
- "precipiced", "despert", "alimned" 같이 영어 사전에 없는 형태의
  "단어"도 만들어내는데, 이는 모델이 단어를 통째로 외운 게 아니라
  글자 단위 패턴(흔한 접두/접미사, 음절 구조)을 학습해서 **새로운
  글자 조합을 생성**하고 있다는 증거이기도 합니다 (동시에 모델
  규모/데이터 양의 한계를 보여주는 부분이기도 합니다).

## 요약

| 항목 | 값 |
|---|---|
| 학습 환경 | Colab, GPU (cuda) |
| 총 학습 스텝 | 3,000 |
| 최종 train loss | 1.2269 |
| 최종 val loss | 1.3844 |
| val perplexity | 3.99 |
| Train/Val 최종 격차 | +0.158 (약한 과적합) |
| 체크포인트 | `gpt2_mini_frankenstein.pt` |
