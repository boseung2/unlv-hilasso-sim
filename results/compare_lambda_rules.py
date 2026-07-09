# λ 선택 규칙 비교 실험 — val-min(현재) vs val-1se(해결법) vs F1-oracle(진단 상한)
#
# 배경: Dataset III·IV에서 F1이 논문 대비 크게 낮음(LASSO III ratio 0.40, IV 0.64).
# 가설: p=10,000에서 held-out validation MSE 최소 λ가 변수를 과다선택 → precision 하락.
# 검증: 같은 glmnet λ 경로 위에서 선택 규칙만 3가지로 바꿔 F1·precision·recall·선택개수 비교.
#   1) val-min  : val MSE 최소 λ (현재 노트북 방식)
#   2) val-1se  : 최소 MSE + 1·SE 이내에서 가장 큰 λ (가장 희소한 해). SE = std(제곱오차)/√n_val
#   3) f1-oracle: 경로상 F1 최대 λ (참 β 사용 — 방법이 아니라 진단용 상한선)
#
# 프로토콜은 notebooks/01_lasso.ipynb §1~7과 동일(단위분산 Σ, β 고정 SEED_BETA=34, glmnet 5.0).
# 실행: cd project_v2 && ../.conda/bin/python results/compare_lambda_rules.py

import os
os.environ["R_HOME"] = "/Users/boseung/Desktop/Lecture/UNLV/.conda/lib/R"

import gc
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

glmnet = importr("glmnet")
_conv = ro.default_converter + numpy2ri.converter

def to_r(a):
    with localconverter(_conv):
        return ro.conversion.get_conversion().py2rpy(np.asarray(a, dtype=float))

# ---------------- 전역 설정 (노트북 §2와 동일) ----------------
N_REPEAT  = 10
SEED_BETA = 34
BASE_SEED = 1000
SIGMA     = 3.0

DATASETS = {
    "Dataset I":  dict(p=100,  groups=[3, 3, 4],   within=0.9, cross=[],
                       n_tr=50,  n_val=10, n_te=10, beta_kind="fixed"),
    "Dataset II": dict(p=1000, groups=[15, 15, 20], within=0.9, cross=[(1, 2, 0.3)],
                       n_tr=100, n_val=20, n_te=20, beta_kind="normal", n_nonzero=50),
    "Dataset III": dict(p=10000, groups=[15, 15, 20], within=0.9, cross=[(1, 2, 0.3)],
                        n_tr=200, n_val=40, n_te=40, beta_kind="normal", n_nonzero=50, big=True),
    "Dataset IV":  dict(p=10000, groups=[15, 15, 20], within=0.9, cross=[(1, 2, 0.3)],
                        n_tr=400, n_val=80, n_te=80, beta_kind="normal", n_nonzero=50, big=True),
}

# ---------------- 공분산·β·데이터 (노트북 §3~5와 동일) ----------------
def build_cov(cfg):
    p = cfg["p"]; groups = cfg["groups"]; within = cfg["within"]
    corr = np.zeros((p, p))
    starts = np.cumsum([0] + list(groups))
    for gi, g in enumerate(groups):
        a, b = starts[gi], starts[gi] + g
        corr[a:b, a:b] = within
    for gi, gj, v in cfg["cross"]:
        ai, bi = starts[gi], starts[gi] + groups[gi]
        aj, bj = starts[gj], starts[gj] + groups[gj]
        corr[ai:bi, aj:bj] = v
        corr[aj:bj, ai:bi] = v
    np.fill_diagonal(corr, 1.0)
    return corr

def corr_block(cfg):
    groups = cfg["groups"]; within = cfg["within"]
    m = int(sum(groups))
    blk = np.zeros((m, m))
    starts = np.cumsum([0] + list(groups))
    for gi, g in enumerate(groups):
        a, b = starts[gi], starts[gi] + g
        blk[a:b, a:b] = within
    for gi, gj, v in cfg["cross"]:
        ai, bi = starts[gi], starts[gi] + groups[gi]
        aj, bj = starts[gj], starts[gj] + groups[gj]
        blk[ai:bi, aj:bj] = v
        blk[aj:bj, ai:bi] = v
    np.fill_diagonal(blk, 1.0)
    return m, blk

FIXED_BETA_I = np.array([3, 3, -3, 2, 2, -2, 1.5, 1.5, 1.5, -1.5], dtype=float)

def build_beta(cfg):
    p = cfg["p"]; beta = np.zeros(p)
    if cfg["beta_kind"] == "fixed":
        beta[:FIXED_BETA_I.size] = FIXED_BETA_I
    else:
        rb = np.random.default_rng(SEED_BETA)
        k = cfg["n_nonzero"]
        beta[:k] = rb.normal(loc=0.0, scale=2.0, size=k)
    return beta

def generate_split(cfg, beta, rep):
    p = cfg["p"]; ntr, nval, nte = cfg["n_tr"], cfg["n_val"], cfg["n_te"]
    n = ntr + nval + nte
    rng = np.random.default_rng(BASE_SEED + rep)
    if cfg.get("big", False):
        m, blk = corr_block(cfg)
        Xc = rng.multivariate_normal(np.zeros(m), blk, size=n)
        Xi = rng.standard_normal((n, p - m))
        X = np.hstack([Xc, Xi])
    else:
        cov = build_cov(cfg)
        X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    eps = rng.normal(0.0, SIGMA, size=n)
    y = X @ beta + eps
    Xtr, Xval, Xte = X[:ntr], X[ntr:ntr + nval], X[ntr + nval:]
    ytr, yval, yte = y[:ntr], y[ntr:ntr + nval], y[ntr + nval:]
    ymean = ytr.mean()
    return (Xtr, ytr - ymean), (Xval, yval - ymean), (Xte, yte - ymean)

# ---------------- 채점 (노트북 §7과 동일) ----------------
def rme(bhat, beta, cfg, sigma=SIGMA):
    m, blk = corr_block(cfg)
    d = bhat - beta
    return float(d[:m] @ blk @ d[:m] + d[m:] @ d[m:]) / (sigma ** 2)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def f1_selection(bhat, beta):
    sel = bhat != 0; true = beta != 0
    tp = int(np.sum(sel & true)); fp = int(np.sum(sel & ~true)); fn = int(np.sum(~sel & true))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(precision=prec, recall=rec, f1=f1, n_selected=int(np.sum(sel)))

# ---------------- glmnet 적합 + 3가지 선택 규칙 ----------------
def fit_and_select(train, val, beta, l1_ratios):
    """같은 λ 경로 위에서 val-min / val-1se / f1-oracle 세 규칙의 해를 모두 반환."""
    Xtr, ytr = train; Xval, yval = val
    n_val = len(yval)
    rXtr = to_r(Xtr); rytr = to_r(ytr.reshape(-1, 1)); rXval = to_r(Xval)

    # l1_ratio(=alpha)별 경로를 전부 계산해 두고 규칙별로 고른다.
    # 주의: np.asarray는 R 메모리의 뷰일 수 있어, del fit + R gc() 뒤에 쓰면
    # 해제된 메모리를 읽는다(간헐적으로 전부 0). 반드시 copy=True로 소유권 있는 사본을 만든다.
    paths = []  # (a, lam, B, mse, errs)
    for a in l1_ratios:
        fit = glmnet.glmnet(rXtr, rytr, alpha=float(a), standardize=True, intercept=False)
        lam = np.array(fit.rx2('lambda'), copy=True).ravel()
        B = np.array(ro.r['as.matrix'](fit.rx2('beta')), copy=True)
        P = np.array(ro.r['predict'](fit, newx=rXval), copy=True)
        if P.ndim == 1:
            P = P[:, None]
        errs = (yval[:, None] - P) ** 2          # n_val x nλ 표본별 제곱오차
        mse = errs.mean(axis=0)
        paths.append((a, lam, B, mse, errs))
        del fit
    del rXtr, rytr, rXval
    gc.collect(); ro.r("gc()")

    out = {}
    # (1) val-min: 모든 alpha·λ 중 val MSE 최소
    best = min(((a, lam, B, mse, errs) for a, lam, B, mse, errs in paths),
               key=lambda t: t[3].min())
    a, lam, B, mse, errs = best
    j = int(mse.argmin())
    out["val-min"] = dict(l1_ratio=a, lam=float(lam[j]), bhat=B[:, j].copy())

    # (2) val-1se: 이긴 alpha의 경로에서, 최소 MSE + 1·SE 이내인 가장 큰 λ
    #     (glmnet의 lam은 내림차순 → 조건을 만족하는 가장 작은 인덱스)
    se = float(errs[:, j].std(ddof=1)) / np.sqrt(n_val)
    j1 = int(np.argmax(mse <= mse[j] + se))      # 첫 True = 가장 큰 λ
    out["val-1se"] = dict(l1_ratio=a, lam=float(lam[j1]), bhat=B[:, j1].copy(), se=se)

    # (3) f1-oracle: 참 β 기준 경로상 F1 최대 (진단용 상한 — 실제 방법 아님)
    best_f1, best_pick = -1.0, None
    for a2, lam2, B2, _, _ in paths:
        for k in range(B2.shape[1]):
            f = f1_selection(B2[:, k], beta)["f1"]
            if f > best_f1:
                best_f1, best_pick = f, (a2, float(lam2[k]), B2[:, k].copy())
    out["f1-oracle"] = dict(l1_ratio=best_pick[0], lam=best_pick[1], bhat=best_pick[2])
    return out

# ---------------- 본 실험 ----------------
def run(method_name, l1_ratios):
    records = []
    for name, cfg in DATASETS.items():
        beta = build_beta(cfg)
        for rep in range(N_REPEAT):
            tr, va, te = generate_split(cfg, beta, rep)
            picks = fit_and_select(tr, va, beta, l1_ratios)
            Xte, yte = te
            for rule, pk in picks.items():
                bhat = pk["bhat"]
                records.append(dict(dataset=name, method=method_name, rule=rule, rep=rep,
                                    lam=pk["lam"], l1_ratio=pk["l1_ratio"],
                                    RME_All=rme(bhat, beta, cfg),
                                    RMSE=rmse(yte, Xte @ bhat),
                                    **f1_selection(bhat, beta)))
        print(f"[{method_name}] {name}: {N_REPEAT} reps done", flush=True)
    return pd.DataFrame(records)

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    dfs = [run("LASSO", [1.0]),
           run("ENet", [round(0.1 * k, 1) for k in range(1, 10)])]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(here, "lambda_rule_comparison.csv"), index=False)

    # 요약: dataset × method × rule 평균
    cols = ["f1", "precision", "recall", "n_selected", "lam", "RME_All", "RMSE"]
    summary = (df.groupby(["method", "dataset", "rule"])[cols]
                 .mean().round(4).reset_index())
    summary.to_csv(os.path.join(here, "lambda_rule_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print()
    print(summary.to_string(index=False))
