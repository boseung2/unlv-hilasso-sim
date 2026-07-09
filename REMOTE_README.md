# Random LASSO 원격 실행 (datax-app-002, R 불필요)

이 서버(JupyterHub, 112코어, conda·R 없음)에서는 **03_random_lasso만** 돌린다.
순수 Python(numpy+sklearn+joblib)이라 `/jupyterhub/jupyterhub-venv/bin/python3`로 바로 실행된다.
(01_lasso·02_enet은 R glmnet이 필요 → 로컬에서 실행.)

무거운 계산만 원격에서 하고, 그래프는 로컬에서 결과 파일로 그린다.

## 1. 클러스터에서 clone (최초 1회)

```bash
cd ~/boseung
git clone https://github.com/boseung2/unlv-hilasso-sim.git
cd unlv-hilasso-sim
```

이후 코드가 업데이트되면 `git pull`만 하면 된다(public repo라 인증 불필요).

## 2. 실행 (클러스터에서)

```bash
cd ~/boseung/unlv-hilasso-sim
PY=/jupyterhub/jupyterhub-venv/bin/python3

# III·IV만 (112코어의 이점이 큰 무거운 것)
nohup $PY run_random_lasso.py --datasets III IV --n_jobs 32 > run.log 2>&1 &
tail -f run.log          # 진행 확인 (Ctrl-C로 보기만 종료, 작업은 계속)

# 또는 4개 전부
$PY run_random_lasso.py --n_jobs 32
```

옵션: `--n_jobs`(코어; 공유서버라 기본 32, 여유 있으면 64), `--reps`(반복, 기본 10),
`--L`(부트스트랩 배수, 기본 30), `--datasets`(I II III IV 또는 이름).

예상 시간(n_jobs=32): III ~6분, IV ~9분. n_jobs=64면 대략 절반.

## 3. 결과 회수 (로컬 Mac 터미널에서 — SSH 경로, 인증 불필요)

결과 CSV·NPZ는 gitignore라 repo에 안 올라간다. SSH로 직접 가져온다:

```bash
rsync -av bottia1@datax-app-002:~/boseung/unlv-hilasso-sim/results_random_lasso.* \
  /Users/boseung/Desktop/Lecture/UNLV/project_v2/results/
```

- `results_random_lasso.csv` — 반복별 지표(F1·AUCPR·RME·RMSE·threshold·n_selected)
- `results_random_lasso.npz` — 그래프용 배열(beta_hats·thresholds·thr곡선·중요도)

이 두 파일을 로컬로 가져오면 그래프(threshold 곡선·분포·PR곡선·계수 산점도)를 로컬에서 그린다.

## 주의

- **공유 서버**: `--n_jobs`를 전체(112)로 두면 다른 사용자에 폐. 기본 32 권장.
- **데이터**: seed로 재생성하므로 서버 내 방법 비교는 정확. 단 numpy/LAPACK 버전차로
  로컬 숫자와 소수점까지 같진 않을 수 있음(논문 비교·경향엔 무관).
- **접속 끊김 대비**: `nohup ... &` 또는 `tmux`로 실행.
