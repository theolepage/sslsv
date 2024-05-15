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
    """
    Compute the error rates for a list of scores and corresponding targets.

    Args:
        scores (List[float]): List of scores.
        targets (List[int]): List of targets (0 for nontarget, 1 for target).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - fprs: False Positive Rates for different score thresholds.
            - fnrs: False Negative Rates for different score thresholds.
            - sorted_scores: Scores sorted in ascending order representing the thresholds.
    """
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
    """
    Compute the Cllr (Log-Likelihood Ratio Cost).

    Args:
        scores (List[float]): List of scores.
        labels (List[int]): List of labels (0 for nontarget, 1 for target).

    Returns:
        float: Cllr value.
    """
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
    """
    Compute the Equal Error Rate (EER).

    Args:
        fprs (numpy.ndarray): Array of False Positive Rates.
        fnrs (numpy.ndarray): Array of False Negative Rates.

    Returns:
        float: EER value.
    """
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
    """
    Compute the minimum Detection Cost Function (minDCF).

    Args:
        fprs (numpy.ndarray): Array of False Positive Rates.
        fnrs (numpy.ndarray): Array of False Negative Rates.
        p_target (float): Prior probability of the target speaker. Defaults to 0.01.
        c_miss (float): Cost associated with a missed detection. Defaults to 1.
        c_fa (float): Cost associated with a false alarm. Defaults to 1.

    Returns:
        float: minDCF value.
    """
    # Equations are from Section 3 of NIST 2016 Speaker Recognition Evaluation Plan

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
    """
    Compute the actual Detection Cost Function (actDCF).

    Args:
        fprs (numpy.ndarray): Array of False Positive Rates.
        fnrs (numpy.ndarray): Array of False Negative Rates.
        sorted_scores (np.ndarray): Array of sorted scores (thresholds).
        p_target (float): Prior probability of the target speaker. Defaults to 0.01.
        c_miss (float): Cost associated with a missed detection. Defaults to 1.
        c_fa (float): Cost associated with a false alarm. Defaults to 1.

    Returns:
        float: actDCF value.
    """
    beta = np.log((c_fa / c_miss) * (1 - p_target) / p_target)
    i = sorted_scores.searchsorted(beta).item()

    c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)

    c_def = min(p_target, 1 - p_target)
    act_dcf = c_det / c_def

    return act_dcf.item()


def avgrprec(trials: Dict[str, Tuple[List[int], List[float]]]) -> float:
    """
    Compute the average R-Precision (avgRPrec).

    Args:
        trials (Dict[str, Tuple[List[int], List[float]]): Dictionary mapping an enrollment ID to a list
            of targets and scores.

    Returns:
        float: avgRPrec value.
    """
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
    """
    Speaker Verification evaluation configuration.

    Attributes:
        num_frames (int): Number of frames to extract from each audio file.
        trials (List[str]): List of paths to trial files.
        metrics (List[str]): List of metrics to compute.
        mindcf_p_target (float): Prior target probability for the minDCF.
        mindcf_c_miss (float): Cost associated with a missed detection for the minDCF. Defaults to 1.
        mindcf_c_fa (float): Cost associated with a false alarm for the minDCF. Defaults to 1.
    """

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
    """
    Speaker Verification (SV) evaluation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Speaker Verification evaluation.

        Args:
            *args: Positional arguments for base class.
            **kwargs: Keyword arguments for base class.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

    def _extract_embeddings_inference(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to perform model inference.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        Y = self.model(X)
        return Y

    def _extract_embeddings_post(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Postprocessing (L2 normalization) applied after model inference.

        Args:
            Y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        Y = F.normalize(Y, p=2, dim=-1)
        return Y

    def _get_sv_score(self, enrol: str, test: str) -> float:
        """
        Get the score for a given enrollment and test trial.

        Args:
            enrol (str): Enrollment ID.
            test (str): Test ID.

        Returns:
            float: Score value.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    def _get_metrics(
        self,
        trials: Dict[str, Tuple[List[int], List[float]]],
        scores: List[float],
        targets: List[int],
        file: Path,
    ) -> Dict[str, float]:
        """
        Determine metrics for a trials file.

        Args:
            trials (Dict[str, Tuple[List[int], List[float]]): Dictionary mapping an enrollment ID
                to a list of targets and scores.
            scores (List[float]): List of scores.
            targets (List[int]): List of targets.
            file (Path): Path to the trials file.

        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
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
        """
        Evaluate on a trials file and return the metrics.

        Args:
            file (Path): Path to the trials file.

        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
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
        """
        Evaluate the model on Speaker Verification (SV).

        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        self._prepare_evaluation()

        metrics = {}
        for file in self.task_config.trials:
            metrics.update(self._evaluate_trials(file))

        return metrics
