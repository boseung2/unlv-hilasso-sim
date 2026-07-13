#!/usr/bin/env python3
"""Hi-LASSO 원격 실행 스크립트 (클러스터용, R 불필요).

numpy + scipy + scikit-learn + joblib. 결과를 CSV(지표) + NPZ(그래프용)로 저장.
선택 = 이항 유의성 검정(임계값 튜닝 없음).

사용 예:
  python3 run_hi_lasso.py --datasets III IV --n_jobs 64
  python3 run_hi_lasso.py                       # 4개 전부
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import numpy as np
import pandas as pd
import generate_data as gd
import scoring
import hi_lasso as hl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Dataset I", "Dataset II", "Dataset III", "Dataset IV"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05, help="유의성 검정 유의수준")
    ap.add_argument("--n_jobs", type=int, default=64, help="병렬 코어(공유서버 배려)")
    ap.add_argument("--out", default="results_hi_lasso")
    args = ap.parse_args()

    names = [d if d.startswith("Dataset") else f"Dataset {d}" for d in args.datasets]
    rows, arrays = [], {}
    print(f"n_jobs={args.n_jobs}, reps={args.reps}, L={args.L}, alpha={args.alpha}, datasets={names}", flush=True)

    for name in names:
        cfg = gd.DATASETS[name]; beta = gd.build_beta(cfg)
        t0 = time.time()
        beta_hats, nsel, imp0, pval0 = [], [], None, None
        for rep in range(args.reps):
            (Xtr, ytr), _, (Xte, yte) = gd.generate_split(cfg, beta, rep)
            r = hl.fit_hi_lasso(Xtr, ytr, L=args.L, alpha=args.alpha,
                                random_state=rep, n_jobs=args.n_jobs)
            bh = r["coef_"]
            rows.append(dict(dataset=name, rep=rep,
                             RME_All=scoring.rme(bh, beta, cfg),
                             RME_Nonzeros=scoring.rme_nonzeros(bh, beta, cfg),
                             RMSE=scoring.rmse(yte, Xte @ bh),
                             aucpr=scoring.aucpr(r["beta_hat_"], beta),
                             **scoring.f1_selection(bh, beta)))
            beta_hats.append(r["beta_hat_"]); nsel.append(r["n_selected"])
            if rep == 0:
                imp0, pval0 = r["importance_"], r["p_values_"]
        arrays[f"{name}__beta"] = beta
        arrays[f"{name}__beta_hats"] = np.array(beta_hats)
        arrays[f"{name}__n_selected"] = np.array(nsel)
        arrays[f"{name}__importance0"] = imp0
        arrays[f"{name}__pvalues0"] = pval0
        print(f"{name}: {args.reps} reps done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out + ".csv", index=False)
    np.savez_compressed(args.out + ".npz", **arrays)
    summ = df.groupby("dataset")[["f1", "aucpr", "RME_All", "RMSE", "n_selected"]] \
             .agg(["mean", "sem"]).round(4)
    print("\n" + summ.to_string(), flush=True)
    print(f"\nsaved: {args.out}.csv + {args.out}.npz", flush=True)


if __name__ == "__main__":
    main()
