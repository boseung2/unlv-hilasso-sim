#!/usr/bin/env python3
"""Hi-LASSO on real GBM/LGG survival data — 유전자 랭킹 + 이항 유의성 검정.

시뮬레이션(run_hi_lasso.py)과의 차이:
  - 정답 β 없음 → F1/RME/AUCPR 채점 없음. 유전자 랭킹 표만 산출.
  - 데이터: 시드 재생성이 아니라 pkl 1개 로드(266×19777).
  - 랭킹 점수 = |beta_hat_| (Procedure-2 최종 계수), 부호 보존.
  - 유의성 = 조교님 recipe p = binom.sf(s, n=t, p=pi), raw/FDR(BH)/Bonferroni 모두 출력.
  - 검열(censoring) 정보 없음 → '검열 미고려' 선형회귀임(리포트에 명시).

사용:
  로컬 스모크 : python run_hi_lasso_realdata.py --subset 2000 --L 2 --n_jobs -1
  원격 본실행 : python run_hi_lasso_realdata.py --L 30 --n_jobs 64
  log-y 민감도: python run_hi_lasso_realdata.py --L 30 --log_y
  시드 안정성 : python run_hi_lasso_realdata.py --L 30 --seeds 0 1 2
"""
import os, sys, time, pickle, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import numpy as np
import pandas as pd
from scipy.stats import binom
import hi_lasso as hl

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKL = os.path.join(HERE, "notebooks", "GBM&LGG_survival_data.pkl")


def load_data(path, subset=None, log_y=False):
    with open(path, "rb") as fr:
        d = pickle.load(fr)
    X = np.asarray(d["X"], float)
    y = np.asarray(d["y"], float).ravel()
    gene = list(d["gene"])
    if subset:                                  # 스모크용: 앞 N개 유전자만
        X, gene = X[:, :subset], gene[:subset]
    if log_y:                                    # 오른쪽 치우침 완화(민감도 분석)
        y = np.log(y)
    return X, y, gene


def ta_pvalues(t, s, pi):
    """조교님 recipe: p = binom.sf(s, n=t, p=pi). t==0(등장 안 함) → p=1."""
    with np.errstate(all="ignore"):
        p = binom.sf(s, n=t, p=pi)
    return np.where(t == 0, 1.0, p)


def run_once(X, y, gene, args, seed, min_t):
    r = hl.fit_hi_lasso(X, y, L=args.L, alpha=args.alpha, correction="fdr",
                        cv=args.cv, random_state=seed, n_jobs=args.n_jobs,
                        max_iter=args.max_iter)
    beta = r["beta_hat_"]
    t, s, pi = r["t_"], r["s_"], r["pi_"]
    pval = ta_pvalues(t, s, pi)                   # 조교님 공식으로 재계산

    # 최소 등장수 필터: t가 너무 작은 유전자는 |beta_hat|·p=0이 우연히 튀므로 제외.
    #   유의성 검정·다중검정 보정은 t>=min_t 유전자들 안에서만 수행(보정 부담도 감소).
    keep = t >= min_t
    idx = np.where(keep)[0]
    sig_raw = keep & (pval < args.alpha)
    sig_fdr = np.zeros(len(gene), bool)
    sig_bonf = np.zeros(len(gene), bool)
    if len(idx):
        sig_fdr[idx] = hl._select_significant(pval[idx], args.alpha, "fdr")
        sig_bonf[idx] = hl._select_significant(pval[idx], args.alpha, "bonferroni")

    df = pd.DataFrame({
        "gene": gene,
        "beta_hat": beta,                          # 부호 포함(보호적/위험 방향)
        "abs_beta": np.abs(beta),
        "importance": r["importance_"],
        "t": t, "s": s, "pi": pi,
        "p_value": pval,
        "pass_min_t": keep,                        # t>=min_t 통과 여부
        "sig_raw": sig_raw,
        "sig_fdr": sig_fdr,
        "sig_bonf": sig_bonf,
    })
    df = df.sort_values("abs_beta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_PKL)
    ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=5000, help="상관 유전자 수렴용")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--subset", type=int, default=None, help="앞 N개 유전자만(스모크)")
    ap.add_argument("--log_y", action="store_true", help="log(y) 민감도 분석")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min_t", type=int, default=-1,
                    help="최소 등장수 필터(기본 -1=L//2). 희귀-운 유전자 배제")
    ap.add_argument("--out", default=os.path.join(HERE, "results_hi_lasso_realdata"))
    args = ap.parse_args()

    min_t = args.min_t if args.min_t >= 0 else max(1, args.L // 2)
    X, y, gene = load_data(args.data, args.subset, args.log_y)
    tag = "log_y" if args.log_y else "raw_y"
    print(f"data: X={X.shape}, y={tag}, L={args.L}, min_t={min_t}, seeds={args.seeds}, "
          f"n_jobs={args.n_jobs}", flush=True)

    tops = {}
    primary = None
    for seed in args.seeds:
        t0 = time.time()
        df = run_once(X, y, gene, args, seed, min_t)
        top_genes = df[df.pass_min_t]["gene"].head(args.topk).tolist()
        tops[seed] = top_genes
        n_raw, n_fdr, n_bonf = df.sig_raw.sum(), df.sig_fdr.sum(), df.sig_bonf.sum()
        print(f"  seed={seed}: {time.time()-t0:.0f}s | pass_min_t={int(df.pass_min_t.sum())}"
              f" | sig raw={n_raw} fdr={n_fdr} bonf={n_bonf}", flush=True)
        if primary is None:
            primary = df

    out_csv = f"{args.out}_{tag}.csv"
    primary.to_csv(out_csv, index=False)

    print(f"\n=== Top {args.topk} genes (seed={args.seeds[0]}, "
          f"t>={min_t}, |beta_hat| 내림차순) ===", flush=True)
    cols = ["rank", "gene", "beta_hat", "t", "s", "p_value", "sig_fdr", "sig_bonf"]
    print(primary[primary.pass_min_t][cols].head(args.topk).to_string(index=False),
          flush=True)

    if len(args.seeds) > 1:                        # 시드 안정성: top-k 겹침
        base = set(tops[args.seeds[0]])
        print(f"\n=== Top-{args.topk} 시드 안정성 (seed {args.seeds[0]} 기준 교집합) ===",
              flush=True)
        for seed in args.seeds[1:]:
            ov = len(base & set(tops[seed]))
            print(f"  seed {seed}: {ov}/{args.topk} 겹침", flush=True)

    print(f"\nsaved: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
