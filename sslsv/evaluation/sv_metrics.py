import numpy as np


def compute_cllr(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)

    target_llrs = scores[labels == 1]
    nontarget_llrs = scores[labels == 0]

    def neglogsigmoid(lodds):
        # -log(sigmoid(x))
        return np.log1p(np.exp(-lodds))

    cllr = 0.5 * (
        np.mean(neglogsigmoid(target_llrs)) +
        np.mean(neglogsigmoid(-nontarget_llrs))
    ) / np.log(2)

    return cllr


def compute_eer(fprs, fnrs):
    idx = np.nanargmin(np.abs(fnrs - fprs))
    eer  = max(fprs[idx], fnrs[idx]) * 100
    return eer


def compute_mindcf(fprs, fnrs, p_target=0.01, c_miss=1, c_fa=1):
    # Equations are from Section 3 of
    # NIST 2016 Speaker Recognition Evaluation Plan

    # Equation (2)
    min_c_det = float('inf')
    min_c_det_idx = None
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_idx = i

    # Equations (3) and (4)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf


def compute_actdcf(fprs, fnrs, sorted_scores, p_target=0.01, c_miss=1, c_fa=1):
    beta = np.log((c_fa / c_miss) * (1 - p_target) / p_target)
    i = sorted_scores.searchsorted(beta).item()

    c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)

    c_def = min(p_target, 1 - p_target)
    act_dcf = c_det / c_def

    return act_dcf


def compute_avgrprec(trials):
    rprec = []

    for e, (targets, scores) in trials.items():
        r = sum(targets)
        if r == 0: continue

        targets_sorted_by_scores = [
            targets[i]
            for i in np.argsort(scores)
        ]
        rprec.append(sum(targets_sorted_by_scores[-r:]) / r)

    avgrprec = np.mean(rprec)
    return avgrprec