from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from pathlib import Path
import numpy as np

from sslsv.evaluations._BaseEvaluation import BaseEvaluation, EvaluationTaskConfig


def compute_error_rates(
    scores: List[float],
    targets: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.array(scores)
    targets = np.array(targets)

    nb_target_scores = len(scores[targets == 1])
    nb_nontarget_scores = len(scores[targets == 0])

    sorted_idx = np.argsort(scores)

    # Determine the number of positives that will be classified
    # as negatives as the score/threshold increases.
    sum_fn = np.cumsum(targets[sorted_idx])

    # Determine the number of negatives that will be classified
    # as positives as the score/threshold increases.
    sum_fp = np.cumsum(np.where(targets[sorted_idx] == 0, 1, 0))

    fnrs = np.empty(len(scores) + 1)
    fnrs[0] = 0
    fnrs[1:] = sum_fn / nb_target_scores

    fprs = np.empty(len(scores) + 1)
    fprs[0] = 1
    fprs[1:] = (nb_nontarget_scores - sum_fp) / nb_nontarget_scores

    return fprs, fnrs, scores[sorted_idx]


def cllr(scores: List[float], labels: List[int]) -> float:
    scores = np.array(scores)
    labels = np.array(labels)

    target_llrs = scores[labels == 1]
    nontarget_llrs = scores[labels == 0]

    def neglogsigmoid(lodds: np.ndarray) -> np.ndarray:
        # -log(sigmoid(x))
        return np.log1p(np.exp(-lodds))

    cllr = (
        0.5
        * (
            np.mean(neglogsigmoid(target_llrs))
            + np.mean(neglogsigmoid(-nontarget_llrs))
        )
        / np.log(2)
    )

    return cllr.item()


def eer(fprs: np.ndarray, fnrs: np.ndarray) -> float:
    idx = np.nanargmin(np.abs(fnrs - fprs))
    eer = max(fprs[idx], fnrs[idx]) * 100
    return eer.item()


def mindcf(
    fprs: np.ndarray,
    fnrs: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1,
    c_fa: float = 1,
) -> float:
    # Equations are from Section 3 of
    # NIST 2016 Speaker Recognition Evaluation Plan

    # Equation (2)
    min_c_det = float("inf")
    min_c_det_idx = None
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_idx = i

    # Equations (3) and (4)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf.item()


def actdcf(
    fprs: np.ndarray,
    fnrs: np.ndarray,
    sorted_scores: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1,
    c_fa: float = 1,
) -> float:
    beta = np.log((c_fa / c_miss) * (1 - p_target) / p_target)
    i = sorted_scores.searchsorted(beta).item()

    c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)

    c_def = min(p_target, 1 - p_target)
    act_dcf = c_det / c_def

    return act_dcf.item()


def avgrprec(trials: Dict[str, Tuple[List[int], List[float]]]) -> float:
    rprec = []

    for e, (targets, scores) in trials.items():
        r = sum(targets)
        if r == 0:
            continue

        targets_sorted_by_scores = [targets[i] for i in np.argsort(scores)]
        rprec.append(sum(targets_sorted_by_scores[-r:]) / r)

    avgrprec = np.mean(rprec)
    return avgrprec.item()


@dataclass
class SpeakerVerificationEvaluationTaskConfig(EvaluationTaskConfig):

    num_frames: int = 10

    trials: List[str] = field(
        default_factory=lambda: [
            "voxceleb1_test_O",
            # 'voxceleb1_test_H',
            # 'voxceleb1_test_E',
            # 'voxsrc2021_val',
            # 'voices2019_dev'
        ]
    )

    metrics: List[str] = field(
        default_factory=lambda: [
            "eer",
            "mindcf",
            # 'actdcf',
            # 'cllr',
            # 'avgrprec'
        ]
    )

    mindcf_p_target: float = 0.01
    mindcf_c_miss: float = 1
    mindcf_c_fa: float = 1


class SpeakerVerificationEvaluation(BaseEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_embeddings_inference(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X)
        return Y

    def _extract_embeddings_post(self, Y: torch.Tensor) -> torch.Tensor:
        Y = F.normalize(Y, p=2, dim=-1)
        return Y

    def _get_sv_score(self, enrol: str, test: str) -> float:
        raise NotImplementedError

    def _get_metrics(
        self,
        trials: Dict[str, Tuple[List[int], List[float]]],
        scores: List[float],
        targets: List[int],
        file: Path,
    ) -> Dict[str, float]:
        metrics = {}

        fprs, fnrs, sorted_scores = compute_error_rates(scores, targets)

        if "eer" in self.task_config.metrics:
            metrics[f"{file}/eer"] = eer(fprs, fnrs)

        if "mindcf" in self.task_config.metrics:
            metrics[f"{file}/mindcf"] = mindcf(
                fprs,
                fnrs,
                p_target=self.task_config.mindcf_p_target,
                c_miss=self.task_config.mindcf_c_miss,
                c_fa=self.task_config.mindcf_c_fa,
            )

        if "actdcf" in self.task_config.metrics:
            metrics[f"{file}/actdcf"] = actdcf(
                fprs,
                fnrs,
                sorted_scores,
                p_target=self.task_config.mindcf_p_target,
                c_miss=self.task_config.mindcf_c_miss,
                c_fa=self.task_config.mindcf_c_fa,
            )

        if "cllr" in self.task_config.metrics:
            metrics[f"{file}/cllr"] = cllr(scores, targets)

        if "avgrprec" in self.task_config.metrics:
            metrics[f"{file}/avgrprec"] = avgrprec(trials)

        return metrics

    def _evaluate_trials(self, file: Path) -> Dict[str, float]:
        trials, scores, targets = {}, [], []

        with open(self.config.dataset.base_path / file) as f:
            lines = f.readlines()
        for line in lines:  # tqdm(lines, desc='Computing scores')
            target, enrol, test = line.rstrip().split(" ")

            score = self._get_sv_score(enrol, test)
            target = int(target)

            scores.append(score)
            targets.append(target)

            if enrol not in trials:
                trials[enrol] = ([], [])
            trials[enrol][0].append(target)
            trials[enrol][1].append(score)

        self.scores = scores
        self.targets = targets

        return self._get_metrics(trials, scores, targets, file)

    def evaluate(self) -> Dict[str, float]:
        self._prepare_evaluation()

        metrics = {}
        for file in self.task_config.trials:
            metrics.update(self._evaluate_trials(file))

        return metrics
