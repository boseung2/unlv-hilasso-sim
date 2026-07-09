#!/usr/bin/env python3
"""Random LASSO 원격 실행 스크립트 (JupyterHub/클러스터용, R 불필요).

numpy + scikit-learn + joblib 만 사용. 결과를 CSV(지표) + NPZ(그래프용 배열)로 저장.
그래프는 로컬에서 이 파일들로 그리면 된다(무거운 계산만 원격).

사용 예:
  python3 run_random_lasso.py                          # 4개 데이터셋, n_jobs=32
  python3 run_random_lasso.py --datasets III IV        # III·IV만 (112코어의 이점)
  python3 run_random_lasso.py --n_jobs 64 --reps 10    # 코어·반복 조절
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")               # sklearn LassoCV 수렴 경고 억제
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import numpy as np
import pandas as pd
import generate_data as gd
import scoring
import random_lasso as rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Dataset I", "Dataset II", "Dataset III", "Dataset IV"],
                    help="이름 또는 로마숫자(I II III IV)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--L", type=int, default=30, help="부트스트랩 배수 B=floor(L*p/n)")
    ap.add_argument("--n_jobs", type=int, default=32, help="병렬 코어(공유서버 예의: 기본 32)")
    ap.add_argument("--out", default="results_random_lasso")
    args = ap.parse_args()

    names = [d if d.startswith("Dataset") else f"Dataset {d}" for d in args.datasets]
    rows, arrays = [], {}
    print(f"n_jobs={args.n_jobs}, reps={args.reps}, L={args.L}, datasets={names}", flush=True)

    for name in names:
        cfg = gd.DATASETS[name]; beta = gd.build_beta(cfg)
        t0 = time.time()
        beta_hats, thresholds, nsel = [], [], []
        thr_grid = thr_mse = imp0 = None
        for rep in range(args.reps):
            (Xtr, ytr), (Xval, yval), (Xte, yte) = gd.generate_split(cfg, beta, rep)
            r = rl.fit_random_lasso(Xtr, ytr, Xval, yval, L=args.L,
                                    random_state=rep, n_jobs=args.n_jobs)
            bh = r["coef_"]
            rows.append(dict(dataset=name, rep=rep,
                             RME_All=scoring.rme(bh, beta, cfg),
                             RME_Nonzeros=scoring.rme_nonzeros(bh, beta, cfg),
                             RMSE=scoring.rmse(yte, Xte @ bh),
                             aucpr=scoring.aucpr(r["beta_hat_"], beta),
                             threshold=r["threshold_"],
                             **scoring.f1_selection(bh, beta)))
            beta_hats.append(r["beta_hat_"]); thresholds.append(r["threshold_"])
            nsel.append(r["n_selected"])
            if rep == 0:                          # 그래프용 rep0 곡선/중요도
                thr_grid, thr_mse, imp0 = r["thr_grid_"], r["thr_mse_"], r["importance_"]
        arrays[f"{name}__beta"] = beta
        arrays[f"{name}__beta_hats"] = np.array(beta_hats)
        arrays[f"{name}__thresholds"] = np.array(thresholds)
        arrays[f"{name}__n_selected"] = np.array(nsel)
        arrays[f"{name}__thr_grid"] = thr_grid
        arrays[f"{name}__thr_mse"] = thr_mse
        arrays[f"{name}__importance0"] = imp0
        print(f"{name}: {args.reps} reps done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out + ".csv", index=False)
    np.savez_compressed(args.out + ".npz", **arrays)

    summ = df.groupby("dataset")[["f1", "aucpr", "RME_All", "RMSE", "n_selected"]] \
             .agg(["mean", "sem"]).round(4)
    print("\n" + summ.to_string(), flush=True)
    print(f"\nsaved: {args.out}.csv (지표) + {args.out}.npz (그래프용 배열)", flush=True)


if __name__ == "__main__":
    main()
