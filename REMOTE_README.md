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

## 2A. 실행 — 노트북 리포트 (그래프까지 원격에서, 권장)

노트북을 통째로 실행해 **표+그래프가 다 든 HTML 리포트**를 원격에 생성한다.
결과는 JupyterHub 웹에서 바로 열어보면 되므로 로컬 회수 불필요.

```bash
cd ~/boseung/unlv-hilasso-sim/notebooks
PY=/jupyterhub/jupyterhub-venv/bin/python3
nohup $PY -m nbconvert --to html --execute 03_random_lasso.ipynb \
  --ExecutePreprocessor.timeout=9000 --ExecutePreprocessor.kernel_name=python3 \
  --output 03_random_lasso_report > run.log 2>&1 &
tail -f run.log
```

→ `notebooks/03_random_lasso_report.html` 생성. **JupyterHub 파일 브라우저에서 열면** 그래프
(threshold 곡선·분포·PR곡선·계수 산점도) + 표(F1·AUCPR·논문비교)가 다 보인다.
`N_JOBS=64`는 노트북 §2 설정에 있음(공유서버 상황 보고 조절).

예상: 4개 전체 ~12분(n_jobs=64). nbconvert가 없으면 `$PY -m pip install --user nbconvert`.

## 2B. 실행 — 지표만 빠르게 (스크립트, 선택)

그래프 없이 숫자만 빠르게 원할 때. CSV+NPZ 저장.

```bash
cd ~/boseung/unlv-hilasso-sim
$PY run_random_lasso.py --datasets III IV --n_jobs 32
```
옵션: `--n_jobs --reps --L --datasets`. 예상: III ~6분, IV ~9분.

## 2C. 실행 — Hi-LASSO (논문 버전)

Hi-LASSO(ElasticNet Proc1 + Adaptive Proc2 + 이항 유의성 검정). 선택 = p<0.05 검정이라
임계값 튜닝 불필요. `run_hi_lasso.py`가 `results_hi_lasso.csv/.npz` 저장.

```bash
cd ~/boseung/unlv-hilasso-sim
PY=/jupyterhub/jupyterhub-venv/bin/python3
nohup $PY run_hi_lasso.py --datasets III IV --n_jobs 64 > run_hi.log 2>&1 &
tail -f run_hi.log
```
옵션: `--n_jobs --reps --L --alpha --datasets`. Proc1이 ElasticNet-CV라 Random보다 무거움
→ 예상(n_jobs=64): III ~15~25분, IV ~25~40분. I·II는 로컬에서도 빠름(수십 초~수 분).

## 3. 결과 보기

- **2A(리포트)**: JupyterHub 웹 파일 브라우저에서 `03_random_lasso_report.html` 열기 → **원격에서 다 봄**(로컬 회수 불필요). 로컬에도 두고 싶으면 rsync로 회수.
- **2B(스크립트)**: `results_random_lasso.csv`(지표)·`.npz`(그래프용 배열)를 rsync로 로컬 회수 후 로컬에서 그래프:
  ```bash
  rsync -av bottia1@datax-app-002:~/boseung/unlv-hilasso-sim/results_random_lasso.* \
    /Users/boseung/Desktop/Lecture/UNLV/project_v2/results/
  ```

## 주의

- **공유 서버**: `--n_jobs`를 전체(112)로 두면 다른 사용자에 폐. 기본 32 권장.
- **데이터**: seed로 재생성하므로 서버 내 방법 비교는 정확. 단 numpy/LAPACK 버전차로
  로컬 숫자와 소수점까지 같진 않을 수 있음(논문 비교·경향엔 무관).
- **접속 끊김 대비**: `nohup ... &` 또는 `tmux`로 실행.
