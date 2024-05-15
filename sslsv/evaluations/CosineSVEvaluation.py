from dataclasses import dataclass
from enum import Enum
from typing import List

import torch

from pathlib import Path
import pandas as pd

from sslsv.evaluations._SpeakerVerificationEvaluation import (
    SpeakerVerificationEvaluation,
    SpeakerVerificationEvaluationTaskConfig,
)


class ScoreNormEnum(Enum):
    """
    Enumeration for score normalization methods for cosine-scoring speaker verification.

    Attributes:
        NONE (None): No score normalization method.
        ZNORM (str): Z-score normalization method.
        TNORM (str): T-score normalization method.
        SNORM (str): S-score normalization method.
    """

    NONE = None
    ZNORM = "z-norm"
    TNORM = "t-norm"
    SNORM = "s-norm"


@dataclass
class CosineSVEvaluationTaskConfig(SpeakerVerificationEvaluationTaskConfig):
    """
    Cosine-based Speaker Verification (SV) evaluation configuration.

    Attributes:
        score_norm (ScoreNormEnum): Type of score normalization to be applied.
        score_norm_cohort_size (int): Size of the cohort used for score normalization.
    """

    score_norm: ScoreNormEnum = ScoreNormEnum.NONE
    score_norm_cohort_size: int = 20000


class CosineSVEvaluation(SpeakerVerificationEvaluation):
    """
    Cosine-based (cosine scoring) Speaker Verification (SV) evaluation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Cosine-based SV evaluation.

        Args:
            *args: Positional arguments for base class.
            **kwargs: Keyword arguments for base class.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

    def _extract_train_embeddings(self):
        """
        Extract embeddings from training data for score normalization.

        Returns:
            None
        """
        df = pd.read_csv(self.config.dataset.base_path / self.config.dataset.train)
        if "Set" in df.columns:
            df = df[df["Set"] == "train"]
        files = df["File"].tolist()

        self.train_embeddings = torch.stack(
            list(
                self._extract_embeddings(
                    files, desc="Extracting train embeddings"
                ).values()
            )
        )

    def _extract_test_embeddings(self, trials: List[Path]):
        """
        Extract embeddings from trials files.

        Args:
            trials (List[Path]): List of trials file paths.

        Returns:
            None
        """
        test_files = list(
            dict.fromkeys(
                [
                    line.rstrip().split()[i]
                    for trial_file in trials
                    for line in open(self.config.dataset.base_path / trial_file)
                    for i in (1, 2)
                ]
            )
        )

        self.test_embeddings = self._extract_embeddings(
            test_files, desc="Extracting test embeddings"
        )

    def _prepare_evaluation(self):
        """
        Prepare evaluation by extracting train and test embeddings.

        Returns:
            None
        """
        if self.task_config.score_norm.value:
            self._extract_train_embeddings()

        self._extract_test_embeddings(self.task_config.trials)

    def _compute_score(self, enrol: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for a given enrollment and test trial.

        Args:
            enrol (torch.Tensor): Tensor of enrollement embedding.
            test (torch.Tensor): Tensor of test embedding.

        Returns:
            torch.Tensor: Score tensor.
        """
        return torch.mean(enrol @ test.T, dim=(-2, -1))

    def _compute_norm_stats(self, enrol: torch.Tensor, test: torch.Tensor):
        """
        Compute the score normalization statistics.

        Args:
            enrol (torch.Tensor): Tensor of enrollement embedding.
            test (torch.Tensor): Tensor of test embedding.

        Returns:
            None
        """
        cohort_size = self.task_config.score_norm_cohort_size

        score_e_c = self._compute_score(self.train_embeddings, enrol)
        score_e_c = torch.topk(score_e_c, k=cohort_size, dim=0)[0]
        self.mean_e_c = torch.mean(score_e_c)
        self.std_e_c = torch.std(score_e_c)

        score_t_c = self._compute_score(self.train_embeddings, test)
        score_t_c = torch.topk(score_t_c, k=cohort_size, dim=0)[0]
        self.mean_t_c = torch.mean(score_t_c)
        self.std_t_c = torch.std(score_t_c)

    def _normalize_score(self, score: torch.Tensor) -> torch.Tensor:
        """
        Normalize the score.

        Args:
            score (torch.Tensor): Input score tensor.

        Returns:
            torch.Tensor: Output score tensor.
        """
        if self.task_config.score_norm == ScoreNormEnum.ZNORM:
            score = (score - self.mean_e_c) / self.std_e_c
        elif self.task_config.score_norm == ScoreNormEnum.TNORM:
            score = (score - self.mean_t_c) / self.std_t_c
        elif self.task_config.score_norm == ScoreNormEnum.SNORM:
            score_e = (score - self.mean_e_c) / self.std_e_c
            score_t = (score - self.mean_t_c) / self.std_t_c
            score = (score_e + score_t) / 2

        return score

    def _get_sv_score(self, enrol: str, test: str) -> float:
        """
        Get the score for a given enrollment and test trial.

        Args:
            enrol (str): Enrollment ID.
            test (str): Test ID.

        Returns:
            float: Score value.
        """
        enrol = self.test_embeddings[enrol]
        test = self.test_embeddings[test]

        if self.task_config.score_norm.value:
            self._compute_norm_stats(enrol, test)

        score = self._compute_score(enrol, test)

        if self.task_config.score_norm.value:
            score = self._normalize_score(score)

        return score.item()
