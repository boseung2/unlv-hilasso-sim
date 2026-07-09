"""glmnet(rpy2) 솔버 — LASSO/Elastic-Net 적합 + validation 기반 (l1_ratio, λ) 선택.

R(rpy2)에 의존하는 유일한 모듈. import 시 rpy2가 R을 로드하므로,
이 모듈을 import 하기 **전에** 환경변수 R_HOME 이 설정돼 있어야 한다
(노트북 §1에서 os.environ["R_HOME"]=... 를 먼저 실행).

generate_data·scoring 에는 의존하지 않는다(데이터 튜플만 받아 적합).
고차원(p=10,000)에서 rpy2/R 메모리 누수를 막기 위해 rep 당 R 변환 1회 + gc.
"""
import gc
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

glmnet = importr("glmnet")
_conv = ro.default_converter + numpy2ri.converter


def version():
    """R glmnet 패키지 버전 문자열."""
    return tuple(ro.r('as.character(packageVersion("glmnet"))'))[0]


def to_r(a):
    """numpy 배열 → R 행렬/벡터."""
    with localconverter(_conv):
        return ro.conversion.get_conversion().py2rpy(np.asarray(a, dtype=float))


def fit_path(rXtr, rytr, l1_ratio):
    fit = glmnet.glmnet(rXtr, rytr, alpha=float(l1_ratio), standardize=True, intercept=False)
    # ⚠️ np.asarray(R객체)는 R 메모리 뷰. curves에 보관 후 gc되면 0으로 읽힐 수 있어 copy 필수.
    lam = np.array(fit.rx2('lambda'), copy=True).ravel()        # λ 경로 (복사)
    B = np.asarray(ro.r['as.matrix'](fit.rx2('beta')))          # p x nλ (bhat는 아래서 .copy())
    return fit, lam, B


def val_mse(fit, rXval, yval):
    P = np.asarray(ro.r['predict'](fit, newx=rXval))            # nval x nλ 예측
    if P.ndim == 1:
        P = P[:, None]
    return np.mean((yval[:, None] - P) ** 2, axis=0)            # λ별 val MSE


def select_model(train, val, l1_ratios):
    """train으로 각 l1_ratio의 λ 경로 적합 → val MSE 최소인 (l1_ratio, λ) 선택.
    반환: (best 딕셔너리, curves={l1_ratio:(lam, mse)})."""
    Xtr, ytr = train; Xval, yval = val
    rXtr = to_r(Xtr); rytr = to_r(ytr.reshape(-1, 1)); rXval = to_r(Xval)   # rep당 R 변환 1회
    best, curves = None, {}
    for a in l1_ratios:
        fit, lam, B = fit_path(rXtr, rytr, a)
        mse = val_mse(fit, rXval, yval)
        curves[a] = (lam, mse)
        j = int(mse.argmin())
        if best is None or mse[j] < best["val_mse"]:
            best = dict(l1_ratio=a, lam=float(lam[j]), idx=j,
                        val_mse=float(mse[j]), bhat=B[:, j].copy())
        del fit
    del rXtr, rytr, rXval                                                   # 큰 R 객체 즉시 해제
    gc.collect(); ro.r("gc()")                                             # 메모리 상한 유지
    return best, curves
