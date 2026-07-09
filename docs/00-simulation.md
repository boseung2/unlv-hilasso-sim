# 00. Hi-LASSO 시뮬레이션 — 설계·프로토콜·재현

> **무엇**: Hi-LASSO 논문(Kim et al., IEEE Access 2019) **Section III 시뮬레이션 연구**의 설계·평가지표·재현 프로토콜을 한곳에 정리한다.
> **자매 문서**: 논문 전문 강독은 [02-hi-lasso-paper.md](02-hi-lasso-paper.md). 실제 실행 코드는 [../notebooks/01_lasso.ipynb](../notebooks/01_lasso.ipynb)(LASSO), [../notebooks/02_enet.ipynb](../notebooks/02_enet.ipynb)(Elastic-Net).
> **이 문서의 성격**: 논문 원문 요약 + **v2에서 확정한 프로토콜** + **논문 문장 ↔ 노트북 코드 대응표** + 현재 재현 결과. 예전 project/ 실험이 "튀던" 원인과 그 교정도 여기 정리한다.

---

## 0. TL;DR (30초 요약)

- **테스트셋** = Dataset I(p=100), II(p=1000). 합성 데이터 `y = Xβ + ε`. 정답 β를 우리가 알기 때문에 "어느 변수가 진짜 중요한지" 안다. (III·IV는 로드맵 항목.)
- **평가지표** = RME_All, RME_Nonzeros(계수 정확도), RMSE(예측), **F1**(변수 선택). 논문 Table 3와 같은 4종.
- **핵심 결정 3가지** — 예전 실험이 튀던 원인을 정면으로 고친 것:
  1. **단위분산 Σ** (Σ = 상관행렬, 대각 1). σ²-스케일 안 씀 → 논문 SNR 재현.
  2. **정답 β 고정** (반복 간 동일, X·ε만 재생성). 시드마다 β를 새로 뽑으면 "매번 다른 문제"가 되어 결과가 튐.
  3. **λ 선택 = held-out validation MSE 최소** (CV 아님, 논문 방식).
- **재현 결과**: LASSO·Elastic-Net 모두 논문 Table 3와 같은 자릿수·근접. 방향성 일치.

---

## 1. 데이터 생성 모형 (논문 식 18)

네 개의 가설 모형을 선형회귀로 생성한다:

```
y = β₁x₁ + β₂x₂ + ··· + βₚxₚ + ε
```

여기서 ε ~ N(0, σ²), **xᵢ ~ N(0, 1)** (단위분산). 정답 β는 주어져 있어 성능 평가가 가능하다.
다중공선성을 위해 공분산 행렬 Σ를 설정하며, 데이터셋은 **동일·반대 부호 양쪽으로 강하게 상관된** 변수를 다수 포함한다.

### ⚠️ 결정 ①: 단위분산 (xᵢ ~ N(0,1))

논문은 각 변수를 **분산 1**로 생성한다. 그래서 상관행렬이 곧 공분산 Σ가 된다.

- **왜 중요한가**: 신호대잡음비 SNR = Var(Xᵀβ)/Var(ε) = βᵀΣβ/σ². 만약 Σ를 σ²·corr(분산 9)로 두면 βᵀΣβ가 9배가 되어 **SNR이 9배 부풀려진다** → 논문이 보고한 SNR(I≈2.9, II≈4.2)과 원리적으로 안 맞는다.
- **v2의 처리**: 노트북 `build_cov()`가 상관행렬을 그대로 Σ로 반환한다. 대각은 1.

```
# notebooks/01_lasso.ipynb — build_cov()
corr[a:b, a:b] = within        # 그룹 내부 0.9
corr[ai:bi, aj:bj] = v         # 그룹 간 0.3 (Dataset II)
np.fill_diagonal(corr, 1.0)
return corr                     # Σ = corr (단위분산)
```

> 📎 이것이 예전 project/ 실험이 논문 표와 안 맞던 원인 중 하나(`simulation.py`가 σ²=9 사용). v2는 단위분산으로 통일했다.

---

## 2. Dataset I~IV 설계

### Table 2 — 시뮬레이션 데이터 요약 (논문)

| 데이터셋 | p | 비0 계수 | 그룹 구조 (상관 0.9) | n_tr / n_val / n_te | q₁,q₂ | B | σ | SNR |
|---|---|---|---|---|---|---|---|---|
| **I** | 100 | 10개 **고정** (3,3,−3,2,2,−2,1.5,1.5,1.5,−1.5) | 3+3+4 | 50/10/10 | 50 | 40 | 3 | 2.9 |
| **II** | 1,000 | 50개 ~ N(0,4) | 15+15+20 (2·3그룹 간 0.3) | 100/20/20 | 100 | 198 | 3 | 4.2 |
| III | 10,000 | 50개 ~ N(0,4) | II와 동일 | 200/40/40 | 200 | 1,000 | 3 | 14.4 |
| IV | 10,000 | 50개 ~ N(0,4) | II와 동일 (샘플 2배) | 400/80/80 | 400 | 500 | 3 | 9.5 |

> `q₁,q₂,B`는 부트스트랩 계열(Random/Hi-LASSO)의 하이퍼파라미터. LASSO·Elastic-Net 단독 재현(현재 v2 범위)에는 쓰이지 않는다. III·IV는 로드맵 항목이라 표에 참고로만 둔다.

### Dataset I — 공분산 (논문 식 20)

앞 10개가 신호. 공분산은 블록 대각:

```
⎛ Σ₃⁰·⁹   0      0      0   ⎞
⎜  0     Σ₃⁰·⁹   0      0   ⎟
⎜  0      0    Σ₄⁰·⁹    0   ⎟
⎝  0      0      0     I₉₀  ⎠
```

Σₖᵛ = 대각 1·비대각 v인 k×k 행렬, Iₖ = k×k 단위행렬. 앞 세 변수·다음 세 변수·다음 네 변수가 각각 상관 0.9 그룹, 나머지 90개는 독립. **그룹 크기 합(3+3+4=10)이 정확히 신호 변수 개수와 일치** — 신호끼리만 강상관이라 LASSO가 그룹에서 하나만 골라버리는(F1이 낮아지는) 상황을 의도적으로 만든 설계.

### Dataset II — 공분산 (논문 식 21)

앞 50개가 신호(N(0,4)에서 추출). 2번째·3번째 그룹 사이에 교차상관 0.3:

```
⎛ Σ₁₅⁰·⁹    0      0      0    ⎞
⎜   0     Σ₁₅⁰·⁹  J₀.₃    0    ⎟
⎜   0     J'₀.₃  Σ₂₀⁰·⁹   0    ⎟
⎝   0      0      0      I₉₅₀  ⎠
```

Jᵥ = 모든 원소가 v인 행렬. 노트북 `DATASETS` 설정의 `cross=[(1, 2, 0.3)]`가 이 J₀.₃ 블록.

### ⚠️ 결정 ②: 정답 β 고정

- **Dataset I**: 논문 고정 벡터 `(3,3,−3,2,2,−2,1.5,1.5,1.5,−1.5)`를 앞 10칸에. 부호가 섞여 있어(β₃,β₆,β₁₀ 음수) 상관 0.9 그룹 안에서 부호가 갈리는 — LASSO가 특히 힘들어하는 구조.
- **Dataset II**: 앞 50개를 N(0,4)=normal(0, 2)에서 **전용 seed(`SEED_BETA=34`)로 한 번만** 뽑아 고정.

```
# notebooks/01_lasso.ipynb — build_beta()
rb = np.random.default_rng(SEED_BETA)     # 데이터 rng와 분리된 전용 rng
beta[:k] = rb.normal(loc=0.0, scale=2.0, size=k)   # N(0,4): 표준편차 2
```

> 📎 **왜 고정하나 + 왜 seed 34인가**: 반복마다 β를 새로 뽑으면 "10번 반복"이 사실 매번 다른 참모형을 푸는 셈이라 F1 표준오차가 부풀려진다(예전 project/ 실험이 튀던 가장 큰 원인). β를 고정하고 X·ε만 바꾸는 것이 올바른 실험 설계. Dataset II의 SNR은 seed에 매우 민감한데(그룹 내 부호 상쇄 때문에 40개 seed 범위에서 2~75까지 요동), seed 34가 SNR≈4.4로 논문 4.2에 가장 가까워 채택했다.

---

## 3. 실험 프로토콜

### 분할과 중심화

n = n_tr + n_val + n_te를 한 번에 뽑아 앞에서부터 train/val/test로 자른다(i.i.d.라 셔플 불필요). y는 **train 평균으로 중심화**(절편 없이 적합하므로), val·test에도 train 평균을 빼서 정보 누출을 막는다.

```
# notebooks/01_lasso.ipynb — generate_split()
ymean = ytr.mean()
return (Xtr, ytr - ymean), (Xval, yval - ymean), (Xte, yte - ymean)
```

X는 모집단 평균이 0인 시뮬레이션이라 따로 중심화하지 않는다. 표준화는 glmnet 내부에 맡긴다(계수는 원 스케일로 반환 → 참 β와 직접 비교 가능).

### ⚠️ 결정 ③: λ 선택 = held-out validation

논문은 λ를 **검증셋 예측오차 최소**로 고른다(CV 아님). glmnet 한 번 적합이 λ 경로(약 70~100개)를 주고, 그 전체에 대해 validation MSE를 계산해 최소를 고른다.

```
# notebooks/01_lasso.ipynb — select_model()
for a in l1_ratios:                       # LASSO=[1.0], ENet=[0.1..0.9]
    fit, lam, B = fit_path(Xtr, ytr, a)
    mse = val_mse(fit, Xval, yval)         # λ별 val MSE (predict가 경로 전체 반환)
    j = int(mse.argmin())
    # (l1_ratio, λ) 조합 전체에서 val MSE 최소를 best로
```

- **LASSO**(`01_lasso.ipynb`): `L1_RATIOS=[1.0]` → 사실상 λ만 탐색.
- **Elastic-Net**(`02_enet.ipynb`): `L1_RATIOS=[0.1..0.9]` → 9개 곡선 × λ경로에서 최적 (l1_ratio, λ) 조합 선택.
- 두 노트북이 `BASE_SEED=1000`을 공유 → **완전히 동일한 데이터**를 봄 → 방법 간 비교가 공정.

### 솔버: R glmnet 5.0 (rpy2)

- python-glmnet은 gfortran 의존성 문제로 사용 불가 → R glmnet을 rpy2로 호출. repo 원본과 동일한 Fortran 코어.
- `os.environ["R_HOME"]`를 **rpy2 import보다 먼저** 프로젝트 로컬 conda R로 지정.
- `standardize=True, intercept=False`. glmnet fit은 R 객체로 유지해야 `rx2`로 lambda 경로·계수 행렬에 접근 가능.
- 실행 환경: 프로젝트 로컬 `.conda/`, 노트북 실행은 nbclient. (환경 상세는 상위 메모리 참조.)

---

## 4. 평가지표 (논문 정의)

### RME (Relative Model Error) — 계수 정확도

```
RME = (β̂ − β)ᵀ Σ (β̂ − β) / σ²
```

단순 유클리드 거리가 아니라 **Σ로 가중한** 거리다. E[(x'β̂ − x'β)²] = (β̂−β)ᵀΣ(β̂−β)이므로, RME는 "계수 오차가 예측오차에 기여하는 양을 잡음 분산 단위로 잰 것" = 초과 예측오차/σ².

- **RME_All**: 모든 예측변수로 계산.
- **RME_Nonzeros**: 정답이 0이 아닌 변수만(Dataset I=앞 10개, II=앞 50개)으로 계산. 고차원에서 대부분 계수가 0이라, "진짜 신호를 얼마나 잘 추정했나"를 F1(선택 여부)보다 세밀하게 본다.

```
# notebooks/01_lasso.ipynb — rme_nonzeros()
nz = beta != 0
d = (bhat - beta)[nz]
S = Sigma[np.ix_(nz, nz)]        # Σ의 신호 부분행렬
return float(d @ S @ d) / (sigma ** 2)
```

### RMSE — 예측 정확도

test 세트의 √mean((y − ŷ)²). 예측 `ŷ = Xte @ β̂` (절편 없음, y 이미 중심화됨).

### F1 — 변수 선택 성능

계수 추정치와 정답으로 혼동행렬 구성:

|  | 방법이 **고름** (β̂ ≠ 0) | 방법이 **버림** (β̂ = 0) |
|---|---|---|
| **정답 = 신호** (β ≠ 0) | TP | FN |
| **정답 = 잡음** (β = 0) | FP | TN |

```
Precision(PPV) = TP/(TP+FP)
Recall(TPR)    = TP/(TP+FN)
F1 = 2·(PPV·TPR)/(PPV+TPR)
```

> 📎 **선택 판정 = β̂ ≠ 0 정확히**: glmnet은 좌표하강으로 **정확한 0**을 만들기 때문에 임계값(TOL) 없이 이렇게 판정해도 된다. sklearn처럼 1e-10 잔여값이 남는 solver였다면 TOL이 필요했을 부분. (논문도 시뮬레이션에서는 유의성 검정이 아니라 이 임계값 방식으로 선택 — [02-hi-lasso-paper.md](02-hi-lasso-paper.md) §II.4 참조.)

---

## 5. 논문 문장 ↔ 노트북 코드 대응표

| 논문 (Section III) | 노트북 함수/변수 | 비고 |
|---|---|---|
| 식(18) `y = Xβ + ε`, xᵢ~N(0,1) | `generate_split()` | 단위분산 |
| 식(20)(21) 공분산 Σ | `build_cov()` | 블록 대각, cross 0.3 |
| 식(19) β 고정 / N(0,4) | `build_beta()` + `SEED_BETA=34` | β 반복 간 고정 |
| train/val/test 분할, y 중심화 | `generate_split()`, `ymean` | train 평균 |
| λ = 검증셋 예측오차 최소 | `select_model()`, `val_mse()` | held-out (CV 아님) |
| LASSO / Elastic-Net | `L1_RATIOS=[1.0]` / `[0.1..0.9]` | glmnet alpha |
| RME_All / RME_Nonzeros | `rme()` / `rme_nonzeros()` | Σ 가중 |
| RMSE | `rmse()` | test 세트 |
| F1 (TP/FP/FN/TN) | `f1_selection()` | β̂≠0 판정 |
| SNR = βᵀΣβ/σ² | `snr()` | 검산용 |
| 10회 반복 | `N_REPEAT=10` | 평균±표준오차 |

---

## 6. 현재 재현 결과 (논문 Table 3 대비)

10회 반복 평균. ratio = ours / paper.

### LASSO ([01_lasso.ipynb](../notebooks/01_lasso.ipynb))

| 데이터셋 | 지표 | ours | paper | ratio |
|---|---|---|---|---|
| I | RME_All | 0.8966 | 0.9727 | 0.92 |
| I | RME_Nonzeros | 0.6684 | 0.8619 | 0.78 |
| I | RMSE | 4.5448 | 3.8003 | 1.20 |
| I | F1 | 0.3526 | 0.4399 | 0.80 |
| II | RME_All | 1.7703 | 2.2148 | 0.80 |
| II | RME_Nonzeros | 1.4483 | 2.1858 | 0.66 |
| II | RMSE | 5.2861 | 5.4900 | 0.96 |
| II | F1 | 0.1418 | 0.1169 | 1.21 |

### Elastic-Net ([02_enet.ipynb](../notebooks/02_enet.ipynb))

| 데이터셋 | 지표 | ours | paper | ratio |
|---|---|---|---|---|
| I | RME_All | 0.8910 | 1.0476 | 0.85 |
| I | RME_Nonzeros | 0.6838 | 0.8186 | 0.84 |
| I | RMSE | 4.5908 | 3.7936 | 1.21 |
| I | F1 | 0.3719 | 0.4238 | 0.88 |
| II | RME_All | 1.8026 | 2.4190 | 0.75 |
| II | RME_Nonzeros | 1.4691 | 2.3822 | 0.62 |
| II | RMSE | 5.2142 | 5.5973 | 0.93 |
| II | F1 | 0.1802 | 0.1578 | 1.14 |

### 해석

- **같은 자릿수·근접**: RME는 논문보다 오히려 낮고(0.62~0.92배), RMSE는 Dataset I에서 1.2배. 이 정도 편차는 자연스러운 범위.
- **왜 차이가 나나**: (1) β 고정(우리) vs 논문의 β 처리 방식 차이 — 특히 seed 34의 β가 "평균적인" β보다 추정하기 쉬운 형상일 수 있어 Dataset II RME가 일관되게 낮음. (2) val=10~20개라 λ 선택이 시끄럽고 test도 10~20개라 RMSE의 rep당 분산이 큼(sem ≈ 0.35).
- **방향성은 논문과 일치**: Dataset II에서 F1이 폭락(LASSO 0.14)하고, **Elastic-Net이 LASSO보다 F1 우세**(II 기준 0.18 vs 0.14). Hi-LASSO 논문이 보여주려는 "기존 방법의 한계" 그림이 그대로 재현됨.

---

## 7. 예전 project/ 실험이 튀던 원인 (교정 완료)

v2는 아래 3대 원인을 처음부터 제거하고 시작했다. (원본 분석: `../project/06-test-protocol.md` §4.)

| 원인 | 예전 (project/) | v2 교정 |
|---|---|---|
| **β가 매 시드마다 바뀜** | Dataset II~IV에서 `default_rng(seed)`로 β 재추출 → 매번 다른 문제 | `SEED_BETA=34` 전용 rng로 1회 고정 |
| **분산 규약 불일치** | `simulation.py`가 σ²=9 → SNR 9배 부풀림 | 단위분산 Σ=corr |
| **파일마다 프로토콜 상이** | simulation/paper_sim/simdata가 서로 다른 규약 | 노트북 하나에 전 과정 통합, 전역 설정 셀에서 관리 |

> 부수 효과: val이 작아 λ 선택이 흔들리는 문제(원인 ②)는 남아 있으나, β 고정으로 rep 간 분산이 크게 줄어 실용적으로 안정적이다. 더 줄이려면 n_val을 키우거나 CV로 전환(로드맵).

---

## 8. Hi-LASSO vs Stochastic LASSO — 같은 데이터, 다른 저울

두 논문(Hi-LASSO 2019 / Stochastic LASSO 2026)은 **Dataset I~IV를 그대로 공유**한다(β·Σ·p·n 동일, 동일 계보 벤치마크). 다른 것은 **평가지표 세트**와 **결과 표현 방식** 두 가지다. 강독본은 [02-hi-lasso-paper.md](02-hi-lasso-paper.md)·[03-stochastic-lasso-paper.md](03-stochastic-lasso-paper.md).

### 8.1 평가지표 차이

| 항목 | Hi-LASSO (2019) | Stochastic LASSO (2026) |
|---|---|---|
| Dataset I~IV 설계 | 동일 | 동일 |
| **Test 셋** | **있음** (n_te = train의 20%) | **없음** — "특징 선택 성능만 평가하므로 시험 데이터 미고려" |
| 선택 지표 | F1 | **F1 + AUCPR** (임계값 없는 평가 추가) |
| 계수 지표 | **RME** = (β̂−β)ᵀ**Σ**(β̂−β)/σ² (Σ 가중) | **RMSE_All / RMSE_Nonzeros** = √mean((β̂−β)²) (단순, Σ 없음) |
| 예측 지표 | **RMSE** (test 세트 y 예측오차) | 없음 (test 셋 없음) |
| 강건성 지표 | 없음 | **Kuncheva Index** (반복 간 선택 일관성) |
| 부트스트랩 설정 | 데이터셋마다 q,B 다름 (Table 2) | **q=n, r=30 통일**, B = p/q×30 |
| 반복 | 10회 | 10회 |
| 추가 실험 | GBM 실데이터 1종 | 18종 TCGA 반시뮬레이션 + GBM n초과 선택 |

> ⚠️ **RMSE 이름 충돌**: 두 논문 모두 "RMSE"를 쓰지만 뜻이 다르다. **Hi-LASSO의 RMSE = 예측오차**(y vs ŷ, test 세트), **Stochastic의 RMSE = 계수 추정오차**(β̂ vs β). Stochastic의 RMSE가 Hi-LASSO의 RME에 대응하되 Σ 가중이 없는 버전이다. 두 논문 결과를 나란히 볼 때 반드시 구분할 것.

### 8.2 결과 표현 방식 차이 (원본 그림·표 확인)

| | Hi-LASSO (IEEE Access) | Stochastic LASSO (Sci Rep) |
|---|---|---|
| **결과 형식** | 전부 **표**(Table 3~9) — 그래프 0개 | 전부 **그림**(Fig 1~6) — 표 0개 |
| 성능 지표 표현 | 평균(표준오차) 숫자, 볼드=최고 | **박스플롯**(10회 반복 분포), 방법=색상 |
| 계수·부호 표현 | Table 5~8: β별 참값·추정치·**+/− 부호 횟수 카운트**(10회 중) | Fig 3: β 인덱스별 **참값(검은 원) + 방법별 추정 산점도**(색 도형) |

- **Hi-LASSO Fig 1(A)/Fig 2** 대응물 = Stochastic의 박스플롯. 같은 F1·계수 지표를 Hi-LASSO는 표 한 칸(평균±SE)으로, Stochastic은 박스로 그려 **변동성·이상치·강건성까지** 한눈에 보이게 했다.
- **Hi-LASSO Table 5~8**(부호 +/− 카운트) = Stochastic **Fig 3**(부호 산점도)의 표 버전. 둘 다 "음수 계수 β₃·β₆·β₁₀를 맞추는가"를 본다. Stochastic Fig 3은 10회 반복의 β̂ 산점을 참값 위에 겹쳐, 부호 실패를 시각적으로 드러낸다.

### 8.3 v2 노트북에의 함의

현재 노트북 셀 12는 **Hi-LASSO 스타일**이다: 지표별 막대그래프(mean ± sem) + β vs β̂ 막대(Dataset I 첫 반복 앞 12개). Stochastic 스타일(박스플롯 + 부호 산점도)로 확장하면 10회 반복의 **강건성**과 **부호 추정 능력**까지 보여줄 수 있다. 방법 수가 2개(LASSO·ENet)뿐이라 박스플롯의 방법 간 비교 이점은 아직 작지만, 부호 산점도(Fig 3형)는 방법 수와 무관하게 유효하다. 구체적 채택 여부·장단점은 로드맵 항목으로 관리한다.

---

## 9. 로드맵

현재 v2 범위는 **LASSO·Elastic-Net 단독 재현(Dataset I·II)**. 다음 단계:

1. **Dataset III·IV 추가** — p=10,000. 공분산은 II와 동일 구조, 잡음 변수는 독립이라 거대 행렬 불필요(블록만 구성). 부트스트랩 계열은 느리므로 반복·방법 조절 필요.
2. **비교 방법 확장** — Adaptive LASSO, Relaxed LASSO, Random LASSO(EE/EA), Recursive LASSO. 논문 Table 3의 전체 열 재현.
3. **Hi-LASSO 구현** — Algorithm 2(2단계 부트스트랩, 결측 처리, 전역 오라클 wⱼ=1/Iⱼ, B 결정 공식). `pip install hi_lasso` 패키지 활용 가능. 최종 목표: 논문 F1 기준값(I=0.478, II=0.229, III=0.515, IV=0.609) 재현.

---

*작성: 2026-07-06. 근거: 02-hi-lasso-paper.md(논문 강독), notebooks/01_lasso.ipynb·02_enet.ipynb(재현 코드), project/06·07(예전 프로토콜 분석 이관).*
