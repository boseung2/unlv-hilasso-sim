"""채점 지표 — RME(Σ-가중 계수오차), RMSE(예측오차), F1(변수선택).

순수 numpy. 공분산 블록·σ 는 generate_data 에서 가져온다(R 불필요).
"""
import numpy as np
from sklearn.metrics import average_precision_score
from generate_data import corr_block, SIGMA


def rme(bhat, beta, cfg, sigma=SIGMA):
    """RME_All = (β̂−β)ᵀ Σ (β̂−β)/σ². Σ=corr_block⊕I 이므로 블록 분해로 계산."""
    m, blk = corr_block(cfg)
    d = bhat - beta
    return float(d[:m] @ blk @ d[:m] + d[m:] @ d[m:]) / (sigma ** 2)


def rme_nonzeros(bhat, beta, cfg, sigma=SIGMA):
    """비영 변수만의 RME. 비영은 모두 앞 m개 안이라 blk 부분행렬만 사용."""
    _, blk = corr_block(cfg)
    nz = np.where(beta != 0)[0]
    d = (bhat - beta)[nz]
    S = blk[np.ix_(nz, nz)]
    return float(d @ S @ d) / (sigma ** 2)


def rmse(y_true, y_pred):
    """test 세트 예측 RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def aucpr(bhat, beta):
    """threshold-free 변수선택 성능 = |β̂| 순위 기반 PR 곡선 아래 면적(average precision).
    임계값을 하나로 안 정하고 모든 컷오프에 대한 precision-recall을 요약. 참 β는 라벨로만 사용."""
    ytrue = (beta != 0).astype(int)
    score = np.abs(bhat)
    if ytrue.sum() == 0 or ytrue.sum() == ytrue.size or score.sum() == 0:
        return 0.0
    return float(average_precision_score(ytrue, score))


def f1_selection(bhat, beta):
    """변수 선택 성능. 선택 판정 = β̂≠0. precision/recall/f1/선택수 반환."""
    sel = bhat != 0; true = beta != 0
    tp = int(np.sum(sel & true)); fp = int(np.sum(sel & ~true)); fn = int(np.sum(~sel & true))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(precision=prec, recall=rec, f1=f1, n_selected=int(np.sum(sel)))
