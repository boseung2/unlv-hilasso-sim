# 최종 리포트 작성 계획 (한글 초안 → 최종 영어)

> 목적: UNLV Project 5 최종 리포트의 뼈대를 한글로 먼저 잡고, 이 문서를 그대로 영어로 옮겨 제출한다.
> 다루는 방법 **4개**: LASSO · Elastic-Net · Random LASSO · Hi-LASSO
> 마지막에 **실제 데이터(TCGA/cBioPortal) 테스트 1회** 추가 예정.
> 리포트 구성은 발표 안내표의 6칸 + 팀원별 기여를 따른다.

---

## 0. 큰 그림 — 리포트는 "6칸 + 기여" 채우기

| # | 칸 | 한 줄 목적 | 분량 감 |
|---|---|---|---|
| 1 | 제목 (Title) | 무엇에 관한 프로젝트인가 | 1줄 |
| 2 | 의의 (Significance) | 왜 하는가 | 불릿 3개 |
| 3 | 문헌 조사 (Literature survey) | 남들은 어떻게 했나 | 논문당 2~3문장 |
| 4 | 방법 (Methods) | 우리가 무엇을 구현했나 | 4방법 각각 |
| 5 | 실험 결과 (Results) | 돌려보니 뭐가 나왔나 | 표·그림 필수 |
| 6 | 참고문헌 (References) | 근거 논문 | 목록 |
| + | 팀원 기여 (Contribution) | 누가 무엇을 했나 | 인당 몇 줄 |

핵심 원칙: **새로 쓰는 게 아니라, 이미 있는 자료를 각 칸에 옮겨 담는다.**

---

## 1. 제목 (Title)

- 후보: "Comparison of LASSO-based Feature Selection Methods for High-Dimensional Data"
- 부제로 "from LASSO to Hi-LASSO" 같은 계보 느낌을 줄 수도 있음.

---

## 2. 프로젝트 의의 (Significance) — 불릿 3개

공식 스펙의 문제의식을 3개로 압축한다.

1. 유전체 데이터는 **특징 수천~수만 개 vs 소표본** → 일반 회귀는 과적합.
2. 특징 간 **강한 다중공선성** → 단순 LASSO는 상관된 변수 중 하나만 뽑고 불안정.
3. 따라서 **안정적으로 진짜 특징을 고르는** LASSO 계열 방법의 비교가 필요.

---

## 3. 문헌 조사 (Literature survey) — 논문당 2~3문장

**이미 강독 완료** → `docs/`에서 각 2~3문장으로 압축만 하면 됨.

| 논문 | 자료 위치 | 담을 핵심 |
|---|---|---|
| LASSO (Tibshirani 1996) | `project/01-lasso-basics.md` | L1 벌점으로 계수를 0으로 → 자동 특징 선택 |
| Elastic-Net (Zou & Hastie 2005) | `docs/`(계보), `project/05-elastic-net-paper.md` | L1+L2 혼합 → 상관 변수 그룹을 함께 선택 |
| Random LASSO (Wang et al. 2011) | `docs/01-random-lasso-paper.md` | 부트스트랩 2단계(생성·선택)로 안정성↑ |
| Hi-LASSO (Kim et al. 2019) | `docs/02-hi-lasso-paper.md` | Random LASSO 개선 — 적응 부트스트랩·전역 오라클·통계적 검정 |

(선택) Stochastic LASSO(2026)는 "최신 동향"으로 1~2문장만 언급 가능. `docs/03-stochastic-lasso-paper.md`.

---

## 4. 방법 (Methods) — 계보 순서로 서술

**서술 전략: "왜 다음 방법이 필요했나"를 이야기로 연결한다.** 4개를 따로 나열하지 말고 발전 흐름으로.

```
LASSO ──(상관 변수에 약함)──▶ Elastic-Net ──(불안정)──▶ Random LASSO ──(개선)──▶ Hi-LASSO
```

각 방법마다 3가지만 적으면 충분:
- **아이디어** (한 문장)
- **어떻게 특징을 고르나** (선택 규칙)
- **한계 / 다음 방법으로 넘어가는 이유**

공통 실험 프로토콜은 여기서 한 번만 정리 (근거: `docs/00-simulation.md`):
- 단위분산 공분산 Σ (상관행렬)
- 정답 β 고정(seed 34), X·ε만 재생성
- λ = held-out validation MSE 최소
- 솔버: R glmnet (rpy2)

구현 파일 대응:
- LASSO → `notebooks/01_lasso.ipynb`
- Elastic-Net → `notebooks/02_enet.ipynb`
- Random LASSO → `lib/random_lasso.py`, `notebooks/03_random_lasso.ipynb`
- Hi-LASSO → `lib/hi_lasso.py`, `notebooks/04_hi_lasso.ipynb`

---

## 5. 실험 결과 (Results) — 표·그림 먼저 고르고 해석 한 줄

**"실험 설정 → 결과 표 → 그림 → 해석" 순서.**

### 5-1. 실험 설정 (Experimental setting)
- Dataset I·II (p 작음) / III·IV (p=10,000) 구성 요약 표.
- 지표: RME_All, RME_Nonzeros, RMSE, F1 (+ Random/Hi는 AUCPR). 10회 반복 평균±표준오차.

### 5-2. 합성 데이터 결과 (메인 표)
- **4방법 × 데이터셋** 비교 표 하나가 리포트의 핵심.
- 자료: `results/`, `results_random_lasso.csv`, 각 노트북 출력.
- 현재 확보 상태:
  - LASSO ✅ / Elastic-Net ✅ (I·II, 논문 방향 일치)
  - Random LASSO ✅ (III·IV 결과 있음)
  - Hi-LASSO ⏳ (구현됨, 결과 정리·검증 필요)
- **할 일: 4방법을 같은 표에 모으고, 논문 Table 3 기준값과 나란히 배치.**

### 5-3. 그림
- F1 / RME 막대그래프 (방법별 비교).
- λ 선택 규칙 비교(있으면): `results/lambda-rule-findings.md`.

### 5-4. 실제 데이터 테스트 (앞으로 할 것)
- 데이터: TCGA 또는 cBioPortal 유전체 (공식 스펙 명시 소스).
- 목표: 합성 데이터에서 가장 좋았던 방법을 실제 데이터에 적용 → 선택된 유전자 수·안정성 비교.
- 정답 β가 없으므로 F1 대신 **선택 안정성 / 재현성 / 생물학적 해석**으로 평가.
- 리포트에는 "예비 결과(preliminary)"로 넣거나, 안 되면 "future work"로 이동.

---

## 6. 해석·분석 + 어려웠던 점/향후 연구 (리포트 본문에 녹임)

- 해석: "복잡한 방법(Hi-LASSO)이 III·IV(고차원)에서 더 유리한가?" 흐름으로.
- 어려웠던 점: 실험이 튀던 3원인(β 재생성 등) → 프로토콜로 해결한 경험.
- 향후: Stochastic LASSO, 실제 데이터 확장.

---

## 7. 참고문헌 (References)

논문 5편 서지정보 나열 (LASSO, Elastic-Net, Random LASSO, Hi-LASSO, +Stochastic LASSO). `paper/` 폴더 원본에서 정확한 서지 추출.

---

## 8. 팀원별 기여 (Contribution) — 별도 15%

- **각자 자기가 한 일을 구체적으로.** (예: "Random LASSO 구현 및 III·IV 실험 담당")
- 팀 구성·역할 분담을 먼저 확인해야 채울 수 있음. ← **확인 필요**

---

## 9. 작성 순서 (쉬운 것 → 어려운 것)

1. 참고문헌 (자료 있음, 5분)
2. 문헌 조사 (강독 압축)
3. 방법 (계보 서술 + 프로토콜)
4. 실험 결과 — **먼저 4방법 통합 표부터 완성** (여기가 병목)
5. 의의 (결과 보고 나면 잘 써짐)
6. 제목 + 기여 + 해석/향후

---

## 10. 남은 실제 작업 (계획 이후 To-Do)

- [ ] Hi-LASSO 결과 정리·검증 → 4방법 통합 비교표 완성
- [ ] Dataset I·II·III·IV 전체에 4방법 채우기 (빈칸 메우기)
- [ ] 실제 데이터(TCGA/cBioPortal) 1회 테스트
- [ ] 팀 역할 분담 확인 → 기여 칸 작성
- [ ] 한글 초안 → 영어 최종 변환

---

*작성: 리포트 계획 v1. 최종 제출은 영어. 이 문서를 섹션별로 영어로 옮기면 리포트 완성.*
