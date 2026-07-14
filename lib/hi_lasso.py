"""Hi-LASSO (Kim et al. 2019) — Random LASSO의 2단계 부트스트랩 + 4개선.

datax-lab hi_lasso.py 를 충실히 재현(논문 버전):
  ① 편향 교정 : 미선택 변수 계수는 0이 아닌 NaN → 중요도에 nanmean 사용.
  ② 정교한 중요도 : I_j = nanmean(|coef|) (Random 의 |mean| 과 다름).
  ③ Adaptive : Procedure 2 에서 중요도 가중 페널티(adaptive LASSO).
  ④ 유의성 검정 : 이항검정 p<alpha 로 변수 선택(임계값·검증셋 불필요).

Procedure 1 : ElasticNet(CV-l1_ratio), 균등 선택.
Procedure 2 : Adaptive LASSO, 중요도 가중 선택.
R 불필요(numpy + scipy + sklearn). random_lasso.py 처럼 독립 모듈.

내부 솔버: python-glmnet 부재로 sklearn 사용(ElasticNetCV/LassoCV).
Adaptive 는 열 스케일링으로 에뮬레이션(X̃ⱼ=Xⱼ/wⱼ 적합 후 계수÷wⱼ).
"""
import math
import warnings
import numpy as np
from scipy.stats import binom
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV
from joblib import Parallel, delayed

L1_GRID = [0.1, 0.5, 0.7, 0.9, 1.0]     # ElasticNet CV 혼합비(glmnet alpha에 대응)


def _standardize(Xsub, ysub):
    Xc = Xsub - Xsub.mean(axis=0)
    xstd = np.sqrt((Xc ** 2).sum(axis=0))
    xstd = np.where(xstd == 0, 1.0, xstd)
    return Xc / xstd, ysub - ysub.mean(), xstd


def _fit_enet(Xs, ys, cv, rs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = ElasticNetCV(l1_ratio=L1_GRID, cv=cv, fit_intercept=False, random_state=rs)
        m.fit(Xs, ys)
    return m.coef_


def _fit_adaptive(Xs, ys, weights, cv, rs):
    """adaptive LASSO: 열을 1/weight로 스케일 → LASSO(1-SE) → 계수 역스케일.
    1-SE 규칙(최소 MSE + 1 SE 안에서 가장 큰 λ)로 glmnet 기본에 가깝게 성기게 뽑는다."""
    Xa = Xs / weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LassoCV(cv=cv, fit_intercept=False, random_state=rs)
        m.fit(Xa, ys)
        mse = m.mse_path_.mean(axis=1)
        se = m.mse_path_.std(axis=1) / np.sqrt(m.mse_path_.shape[1])
        imin = int(np.argmin(mse))
        ok = np.where(mse <= mse[imin] + se[imin])[0]
        alpha = m.alphas_[ok[0]] if len(ok) else m.alpha_   # alphas_ 내림차순 → 가장 큰 λ
        m = Lasso(alpha=alpha, fit_intercept=False)
        m.fit(Xa, ys)
    return m.coef_ / weights


def _one_bootstrap(b, X, y, q, select_prob, mode, penalty_weights, cv, random_state):
    rng = np.random.default_rng([random_state, b])
    n, p = X.shape
    rows = rng.integers(0, n, size=n)
    cols = rng.choice(p, size=q, replace=False, p=select_prob)
    Xs, ys, xstd = _standardize(X[rows][:, cols], y[rows])
    if mode == "procedure1":
        coef = _fit_enet(Xs, ys, cv, random_state)
    else:
        coef = _fit_adaptive(Xs, ys, penalty_weights[cols], cv, random_state)
    beta = np.full(p, np.nan)                     # ① 미선택 = NaN (편향 교정)
    beta[cols] = coef / xstd
    return beta


def _bootstrap_stage(X, y, q, B, select_prob, mode, penalty_weights, cv, random_state, n_jobs):
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_bootstrap)(b, X, y, q, select_prob, mode, penalty_weights, cv, random_state)
        for b in range(B))
    betas = np.full((X.shape[1], B), np.nan)
    for b, beta in enumerate(results):
        betas[:, b] = beta
    return betas


def fit_hi_lasso(X, y, q1="auto", q2="auto", L=30, alpha=0.05,
                 correction="bonferroni", cv=5, random_state=0, n_jobs=-1):
    """Hi-LASSO 적합. 반환: dict(coef_, beta_hat_, importance_, p_values_, n_selected).
    선택 = 이항 유의성 검정 — 임계값 튜닝 불필요.
      q="auto" → n//2 (부분문제를 잘 정의: 표본>변수 → LASSO 성김).
      correction: "bonferroni"(기본, p개 동시검정 FWER 보정) | "none" | "fdr"(BH)."""
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    q1 = max(1, n // 2) if q1 == "auto" else int(q1)   # q<n: 포화 회귀 방지
    q2 = max(1, n // 2) if q2 == "auto" else int(q2)

    # Procedure 1 — 중요도 (ElasticNet, 균등 선택)
    B1 = math.floor(L * p / q1)
    b1 = _bootstrap_stage(X, y, q1, B1, None, "procedure1", None, cv, random_state, n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN 행 경고 억제
        b1_mean = np.nanmean(np.abs(b1), axis=1)            # ② nanmean(|coef|)
    importance = np.where(np.isnan(b1_mean) | (b1_mean == 0), 1e-10, b1_mean)
    select_prob = importance / importance.sum()
    penalty_weights = 1.0 / (select_prob * 100)             # ③ adaptive 페널티

    # Procedure 2 — 계수 + 검정 (Adaptive LASSO, 가중 선택)
    B2 = math.floor(L * p / q2)
    b2 = _bootstrap_stage(X, y, q2, B2, select_prob, "procedure2", penalty_weights,
                          cv, random_state, n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b2_mean = np.nan_to_num(np.nanmean(b2, axis=1))     # 평균 계수 (미선택 0)

    # ④ 이항 유의성 검정
    #   각 변수는 부트스트랩에서 q2/p 확률로만 '등장'(평균 ~L회)하므로,
    #   이항검정 시행수 n 은 전체 B2 가 아니라 변수별 등장수 n_j 여야 한다.
    #   (n=B2 로 넣으면 귀무 기대선택수 pi·B2 가 뻥튀기돼 전부 기각됨 — III F1=0 버그.)
    not_null = ~np.isnan(b2)
    n_j = not_null.sum(axis=1)                              # 변수별 등장수(=시행수)
    d = np.logical_and(not_null, b2 != 0).sum(axis=1)       # nonzero 선택 횟수
    pi = d.sum() / max(int(not_null.sum()), 1)              # 귀무 평균 선택률
    p_values = binom.sf(d - 1, n=n_j, p=pi)                 # 등장한 적 없으면 n_j=0 → p=1

    selected = _select_significant(p_values, alpha, correction)
    coef_ = np.where(selected, b2_mean, 0.0)

    return dict(coef_=coef_, beta_hat_=b2_mean, importance_=importance,
                p_values_=p_values, n_selected=int((coef_ != 0).sum()))


def _select_significant(p_values, alpha, correction):
    """다중검정 보정 후 유의 변수 마스크. p개 동시검정이라 보정이 기본."""
    p = len(p_values)
    if correction == "bonferroni":
        return p_values < alpha / p
    if correction == "fdr":                                 # Benjamini-Hochberg
        order = np.argsort(p_values)
        thr = np.arange(1, p + 1) / p * alpha
        passed = p_values[order] <= thr
        sel = np.zeros(p, bool)
        if passed.any():
            sel[order[:np.where(passed)[0].max() + 1]] = True
        return sel
    return p_values < alpha                                 # "none"
