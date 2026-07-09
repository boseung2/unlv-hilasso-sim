# project_v2 — Hi-LASSO 시뮬레이션 재현

UNLV ML 인턴십 프로젝트 5(LASSO 특징 선택)의 **깨끗한 재출발** 폴더.
Hi-LASSO 논문(Kim et al., *IEEE Access* 2019) **Section III 시뮬레이션 연구**를 처음부터 재현한다.
예전 `../project/` 폴더는 여러 실험 파일이 섞여 데이터·프로토콜이 어긋났기 때문에, 여기서 **논문 → 문서 → 코드** 순서로 다시 정리한다.

## 폴더 지도

```
project_v2/
├── README.md                    ← 지금 이 문서 (지도 + 확정 프로토콜 + 로드맵)
├── docs/
│   ├── 00-simulation.md         시뮬레이션 설계·프로토콜·재현 결과 + 코드 대응표 (공통 기반)
│   ├── 01-random-lasso-paper.md  Random LASSO 논문 강독 (2011, Hi-LASSO의 직접 전신)
│   ├── 02-hi-lasso-paper.md     Hi-LASSO 논문 강독 (2019, 이 프로젝트가 재현하는 논문)
│   └── 03-stochastic-lasso-paper.md  Stochastic LASSO 논문 강독 (2026, 최신 SOTA·동일 시뮬 계보)
├── notebooks/
│   ├── 01_lasso.ipynb           LASSO 재현 (glmnet alpha=1)
│   └── 02_enet.ipynb            Elastic-Net 재현 (glmnet alpha=0.1..0.9)
└── results/
    ├── compare_lambda_rules.py  λ 선택 규칙 비교 실험 (val-min/1se/oracle)
    └── lambda-rule-findings.md  III·IV F1 저하 원인 분석 결과
```

**읽는 순서**: [docs/02-hi-lasso-paper.md](docs/02-hi-lasso-paper.md) (논문 이해) → [docs/00-simulation.md](docs/00-simulation.md) (설계·프로토콜) → [notebooks/01_lasso.ipynb](notebooks/01_lasso.ipynb) (코드로 확인). 계보를 거슬러 배경을 보려면 [docs/01-random-lasso-paper.md](docs/01-random-lasso-paper.md) (Hi-LASSO의 전신).

## 확정 프로토콜 (핵심 3가지)

예전 실험이 "튀던" 원인을 처음부터 제거한 세 결정. 근거는 [docs/00-simulation.md](docs/00-simulation.md) §2·§3·§7.

1. **단위분산 Σ** — 공분산 = 상관행렬(대각 1). σ²-스케일 안 씀 → 논문 SNR(I≈2.9, II≈4.2) 재현.
2. **정답 β 고정** — 반복 간 동일(`SEED_BETA=34`), X·ε만 재생성. 시드마다 β를 새로 뽑으면 매번 다른 문제가 되어 결과가 튐.
3. **λ = held-out validation MSE 최소** — CV 아님, 논문 방식.

솔버는 **R glmnet 5.0** (rpy2 경유, repo 원본과 동일한 Fortran 코어). 파이썬은 프로젝트 로컬 `.conda/`.

## 현재 재현 상태

| 방법 | 노트북 | Dataset I·II | 논문 Table 3 대비 |
|---|---|---|---|
| LASSO | `01_lasso.ipynb` | ✅ 완료 | 같은 자릿수·근접, 방향성 일치 |
| Elastic-Net | `02_enet.ipynb` | ✅ 완료 | 같은 자릿수·근접, II에서 F1 > LASSO (논문 방향) |

지표: RME_All, RME_Nonzeros, RMSE, F1 (10회 반복 평균±표준오차). 상세 표는 [docs/00-simulation.md](docs/00-simulation.md) §6.

## 로드맵

1. **Dataset III·IV 추가** (p=10,000) — 공분산은 II와 동일 구조.
2. **비교 방법 확장** — Adaptive / Relaxed / Random(EE·EA) / Recursive LASSO. Random LASSO는 강독 완료([docs/01-random-lasso-paper.md](docs/01-random-lasso-paper.md), 구현 준비 메모 포함).
3. **Hi-LASSO 구현** — Algorithm 2 (2단계 부트스트랩·전역 오라클). 목표: 논문 F1 기준값(I=0.478, II=0.229, III=0.515, IV=0.609) 재현.
4. **Stochastic LASSO 구현** — 최신 SOTA(CBB·Forward Selection·2단계 t-검정). 강독본: [docs/03-stochastic-lasso-paper.md](docs/03-stochastic-lasso-paper.md). 단, 평가지표가 Hi-LASSO와 다름(AUCPR·Kuncheva Index 추가, test 셋 없음) — 비교 시 지표 정렬 필요.

## 재현 방법

```bash
# 프로젝트 로컬 conda 파이썬으로 노트북 실행 (nbclient)
../.conda/bin/python -c "import nbformat; from nbclient import NotebookClient; \
  nb=nbformat.read('notebooks/01_lasso.ipynb',as_version=4); \
  NotebookClient(nb,kernel_name='python3').execute(); \
  nbformat.write(nb,'notebooks/01_lasso.ipynb')"
```

각 노트북은 자립형 — 데이터 생성 → glmnet 적합 → 채점 → 시각화까지 한 파일에 담겨 있다.

---

*마지막 정리: 2026-07-06. 예전 project/ 폴더의 09(논문 강독)·07(실험 방법론)·06(테스트 프로토콜)을 이관·통합.*
