"""project_v2 시뮬레이션 데이터 생성 (Hi-LASSO 논문 Section 3, Dataset I~IV).

이 모듈은 **데이터 생성만** 담당한다(채점·솔버는 각 노트북에 유지).
순수 numpy라 R(rpy2)이 필요 없다.

생성은 seed로 완전히 결정론적이다:
  - 참계수 β 는 SEED_BETA(고정)로 한 번 생성 → 반복과 무관하게 동일.
  - 반복 r 의 데이터(X·ε·분할)는 seed = BASE_SEED + r 로 생성.
따라서 배열을 저장하지 않고 manifest(cfg + seed)만 저장해두면,
같은 환경에서 언제든 동일한 X·y 를 재생성할 수 있다(manifest 방식).

주의: multivariate_normal 은 내부 SVD 가 LAPACK 구현에 의존하므로,
서로 다른 기계에서 재생성한 X 는 미세하게 다를 수 있다(같은 기계 내에서는 동일).
"""
import json
import os
import numpy as np

# ---- 전역 상수 (seed / 잡음) ----
SEED_BETA = 34      # β 고정 seed. Dataset II SNR≈4.4 (논문 4.2)에 맞춘 값
BASE_SEED = 1000    # 반복 r 의 데이터 seed = BASE_SEED + r
SIGMA     = 3.0     # 잡음 표준편차 σ

# ---- 데이터셋 정의 ----
# cross = [(그룹i, 그룹j, 상관값)] (그룹 인덱스 0부터). big=True 면 고차원 블록 경로.
DATASETS = {
    "Dataset I":   dict(p=100,   groups=[3, 3, 4],    within=0.9, cross=[],
                        n_tr=50,  n_val=10, n_te=10, beta_kind="fixed"),
    "Dataset II":  dict(p=1000,  groups=[15, 15, 20], within=0.9, cross=[(1, 2, 0.3)],
                        n_tr=100, n_val=20, n_te=20, beta_kind="normal", n_nonzero=50),
    "Dataset III": dict(p=10000, groups=[15, 15, 20], within=0.9, cross=[(1, 2, 0.3)],
                        n_tr=200, n_val=40, n_te=40, beta_kind="normal", n_nonzero=50, big=True),
    "Dataset IV":  dict(p=10000, groups=[15, 15, 20], within=0.9, cross=[(1, 2, 0.3)],
                        n_tr=400, n_val=80, n_te=80, beta_kind="normal", n_nonzero=50, big=True),
}

FIXED_BETA_I = np.array([3, 3, -3, 2, 2, -2, 1.5, 1.5, 1.5, -1.5], dtype=float)


# ---- 공분산 ----
def build_cov(cfg):
    """전체 p×p 밀집 공분산 Σ=corr (단위분산). 저차원 I·II 데이터 생성용."""
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
    """상관이 있는 앞 m=Σgroups개만의 블록. 나머지 p-m 변수는 독립.
    전체 공분산 Σ = corr_block ⊕ I_{p-m} 이므로 p=10,000에서도 거대 행렬을 안 만든다.
    채점(rme)에서도 재사용한다. 반환: (m, blk)."""
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


# ---- 참계수 β (반복과 무관하게 고정) ----
def build_beta(cfg):
    p = cfg["p"]; beta = np.zeros(p)
    if cfg["beta_kind"] == "fixed":
        beta[:FIXED_BETA_I.size] = FIXED_BETA_I
    else:
        rb = np.random.default_rng(SEED_BETA)          # β 고정용 별도 seed
        k = cfg["n_nonzero"]
        beta[:k] = rb.normal(loc=0.0, scale=2.0, size=k)  # N(0,4)
    return beta


# ---- 데이터 생성 & 분할 ----
def generate_split(cfg, beta, rep):
    """반복 rep 의 데이터 한 벌을 seed=BASE_SEED+rep 로 생성해 train/val/test 로 분할.
    반환: (Xtr, ytr-중심화), (Xval, yval-중심화), (Xte, yte-중심화)."""
    p = cfg["p"]; ntr, nval, nte = cfg["n_tr"], cfg["n_val"], cfg["n_te"]
    n = ntr + nval + nte
    rng = np.random.default_rng(BASE_SEED + rep)
    if cfg.get("big", False):                           # 고차원: 앞 블록만 상관, 나머지 독립
        m, blk = corr_block(cfg)
        Xc = rng.multivariate_normal(np.zeros(m), blk, size=n)
        Xi = rng.standard_normal((n, p - m))
        X = np.hstack([Xc, Xi])
    else:                                               # 저차원(I·II): 밀집 경로
        cov = build_cov(cfg)
        X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    eps = rng.normal(0.0, SIGMA, size=n)
    y = X @ beta + eps
    Xtr, Xval, Xte = X[:ntr], X[ntr:ntr + nval], X[ntr + nval:]
    ytr, yval, yte = y[:ntr], y[ntr:ntr + nval], y[ntr + nval:]
    ymean = ytr.mean()                                  # train 기준 중심화
    return (Xtr, ytr - ymean), (Xval, yval - ymean), (Xte, yte - ymean)


def snr(cfg, beta, sigma=SIGMA):
    """검산용 신호대잡음비 = βᵀΣβ/σ² = β[:m]ᵀblk β[:m] + β[m:]ᵀβ[m:] (β[m:]=0)."""
    m, blk = corr_block(cfg)
    return float(beta[:m] @ blk @ beta[:m] + beta[m:] @ beta[m:]) / (sigma ** 2)


# ---- 편의 함수 ----
def get_split(name, rep):
    """데이터셋 이름 + 반복 번호로 (train, val, test) 재생성."""
    cfg = DATASETS[name]
    return generate_split(cfg, build_beta(cfg), rep)


# ---- manifest (배열 대신 cfg+seed 저장) ----
def write_manifest(path, n_repeat=10):
    """manifest 방식: 실제 배열은 저장하지 않고 재현에 필요한 cfg+seed만 기록."""
    manifest = dict(
        method="manifest (배열 미저장, seed로 재생성)",
        seeds=dict(SEED_BETA=SEED_BETA, BASE_SEED=BASE_SEED,
                   note="반복 r 데이터 seed = BASE_SEED + r; β seed=SEED_BETA(고정)"),
        sigma=SIGMA,
        n_repeat=n_repeat,
        datasets=DATASETS,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", "manifest.json")
    p = write_manifest(out)
    print("manifest written:", os.path.normpath(p))
    for name in DATASETS:
        cfg = DATASETS[name]
        b = build_beta(cfg)
        print("  %-12s p=%-6d n_tr=%-4d SNR=%.3f" % (name, cfg["p"], cfg["n_tr"], snr(cfg, b)))
