"""Random LASSO (Wang et al. 2011) — 2단계 부트스트랩 특징 선택.

datax-lab Hi-LASSO 구조(hi_lasso.py)를 기반으로 하되 **Random LASSO**로 구성:
  - Procedure 1·2 모두 순수 LASSO. 내부 λ 는 5-fold CV(sklearn LassoCV) —
    논문 Algorithm 1은 "LASSO 적용"만 명시, datax-lab glmnet_model이 n_splits=5.
  - 중요도 I_j = |mean(signed coef)| (논문 Algorithm 1, 식 196; 미선택=0).
  - 최종 선택 = 선택빈도 threshold(Proc2에서 eligible 대비 nonzero 비율).

R 불필요(numpy + sklearn). solver.py 처럼 독립 모듈(generate_data·scoring 의존 안 함).

기본 하이퍼파라미터(논문): q1=q2=n, L=30 → B = floor(L·p/q).
"""
import math
import numpy as np
from sklearn.linear_model import LassoCV
from joblib import Parallel, delayed


def _standardize(X, y):
    """열별 중심화 + L2 정규화, y 중심화. coef 역스케일용 열 노름 반환."""
    Xc = X - X.mean(axis=0)
    xnorm = np.sqrt((Xc ** 2).sum(axis=0))
    xnorm = np.where(xnorm == 0, 1.0, xnorm)
    return Xc / xnorm, y - y.mean(), xnorm


def _fit_lasso(Xsub, ysub, cv, random_state):
    """소형 LASSO 한 번: 표준화 → LassoCV(5-fold) → 원 스케일 계수."""
    Xs, ys, xnorm = _standardize(Xsub, ysub)
    m = LassoCV(cv=cv, fit_intercept=False, random_state=random_state)
    m.fit(Xs, ys)
    return m.coef_ / xnorm


def _one_bootstrap(b, X, y, q, select_prob, cv, random_state):
    """부트스트랩 하나. 시드는 [random_state, b]로 독립 → 병렬이어도 재현."""
    rng = np.random.default_rng([random_state, b])
    n, p = X.shape
    rows = rng.integers(0, n, size=n)                              # 표본 복원추출
    cols = rng.choice(p, size=q, replace=False, p=select_prob)     # 예측변수 선택
    coef = _fit_lasso(X[rows][:, cols], y[rows], cv, random_state)
    return cols, coef


def _bootstrap_stage(X, y, q, B, select_prob, cv, random_state, n_jobs):
    """B개 부트스트랩을 joblib으로 병렬 실행. p×B 계수 + eligible 카운트."""
    n, p = X.shape
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_bootstrap)(b, X, y, q, select_prob, cv, random_state)
        for b in range(B))
    coefs = np.zeros((p, B))
    eligible = np.zeros(p, dtype=int)
    for b, (cols, coef) in enumerate(results):
        coefs[cols, b] = coef
        eligible[cols] += 1
    return coefs, eligible


def _select_threshold(beta_hat, Xval, yval, n_grid=40):
    """검증셋 예측오차(MSE)를 최소화하는 |β̂| 임계값 t*를 데이터로 선택(정답 β 미사용).
    01/02가 λ를 검증으로 고르는 것과 동일한 원리. 반환: (t*, 후보격자, 격자별 MSE)."""
    absb = np.abs(beta_hat)
    nz = absb[absb > 0]
    yval = np.asarray(yval, float).ravel()
    if nz.size == 0:
        return 0.0, np.array([0.0]), np.array([float(np.mean(yval ** 2))])
    cand = np.unique(np.concatenate([[0.0], np.quantile(nz, np.linspace(0, 1, n_grid))]))
    Xval = np.asarray(Xval, float)
    mses = np.array([float(np.mean((yval - Xval @ np.where(absb >= t, beta_hat, 0.0)) ** 2))
                     for t in cand])
    return float(cand[mses.argmin()]), cand, mses


def fit_random_lasso(X, y, Xval=None, yval=None, q1="auto", q2="auto", L=30,
                     threshold=None, cv=5, random_state=0, n_jobs=-1):
    """Random LASSO 적합. 반환: dict(coef_, importance_, sel_freq_, n_selected, threshold_).
    threshold 선택: (Xval,yval) 주면 검증 MSE 최소로 자동(정직), 아니면 고정 threshold 사용.
    n_jobs=-1: 부트스트랩을 전체 코어로 병렬(재현성은 부트스트랩별 독립 시드로 보장)."""
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    q1 = n if q1 == "auto" else int(q1)
    q2 = n if q2 == "auto" else int(q2)

    # Procedure 1 — 중요도 산출 (균등 선택)
    B1 = math.floor(L * p / q1)
    coefs1, _ = _bootstrap_stage(X, y, q1, B1, None, cv, random_state, n_jobs)
    importance = np.abs(coefs1.mean(axis=1))                       # |mean signed| (미선택=0)
    imp_floor = np.where(importance == 0, 1e-10, importance)
    select_prob = imp_floor / imp_floor.sum()

    # Procedure 2 — 계수 추정 + 선택 (중요도 가중 선택)
    B2 = math.floor(L * p / q2)
    coefs2, eligible = _bootstrap_stage(X, y, q2, B2, select_prob, cv, random_state, n_jobs)
    beta_hat = coefs2.mean(axis=1)                                 # 평균 계수 (식 207)
    d = (coefs2 != 0).sum(axis=1)                                  # nonzero 선택 횟수
    sel_freq = d / B2                                              # 참고용 선택빈도

    # 최종 선택 임계값: 검증셋이 있으면 검증 MSE 최소로 데이터 선택(정직), 없으면 고정값.
    thr_grid = thr_mse = None
    if Xval is not None and yval is not None:
        threshold, thr_grid, thr_mse = _select_threshold(beta_hat, Xval, yval)
    elif threshold is None:
        threshold = 0.0
    selected = np.abs(beta_hat) >= threshold                       # |β̂| 크기 임계 (논문 식 2.e)
    coef_ = np.where(selected, beta_hat, 0.0)

    return dict(coef_=coef_, beta_hat_=beta_hat, importance_=importance,
                sel_freq_=sel_freq, n_selected=int(selected.sum()), threshold_=float(threshold),
                thr_grid_=thr_grid, thr_mse_=thr_mse)
