from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from sslsv.evaluations._SpeakerVerificationEvaluation import (
    SpeakerVerificationEvaluation,
    SpeakerVerificationEvaluationTaskConfig,
)

from sslsv.utils.distributed import is_main_process


class ScoreNormEnum(Enum):
    """
    Enumeration for score normalization methods for cosine-scoring speaker verification.

    Attributes:
        NONE (None): No score normalization method.
        ZNORM (str): Z-score normalization method.
        TNORM (str): T-score normalization method.
        SNORM (str): S-score normalization method.
        ASNORM (str): AS-score normalization method.
    """

    NONE   = None
    ZNORM  = "z-norm"
    TNORM  = "t-norm"
    SNORM  = "s-norm"
    ASNORM = "as-norm"


@dataclass
class CosineSVEvaluationTaskConfig(SpeakerVerificationEvaluationTaskConfig):
    """
    Cosine-based Speaker Verification (SV) evaluation configuration.

    Attributes:
        score_norm (ScoreNormEnum): Type of score normalization to be applied.
        score_norm_cohort_size (int): Size of the cohort used for score normalization.
    """

    score_norm: ScoreNormEnum = ScoreNormEnum.NONE
    score_norm_cohort_size: int = 300


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

        self.task_config.__subtype__ = self.task_config.score_norm.value

    def _extract_trials_embeddings(self, trials: List[Path]):
        """
        Extract enrollment and test embeddings from trials files.

        Args:
            trials (List[Path]): List of trials file paths.

        Returns:
            None
        """
        def _get_trials_files(row: int) -> List[str]:
            return list(
                dict.fromkeys(
                    [
                        line.rstrip().split()[row]
                        for trial_file in trials
                        for line in open(self.config.dataset.base_path / trial_file)
                    ]
                )
            )

        embeddings_file = self.config.model_path / f"trials_embeddings.pt"

        enrol_files = _get_trials_files(1)
        test_files = _get_trials_files(2)

        # if embeddings_file.exists():
        #     embeddings = torch.load(embeddings_file)

        all_files = list(dict.fromkeys(enrol_files + test_files))
        embeddings = self._extract_embeddings(
            all_files,
            desc="Extracting enrollment and test embeddings"
        )
        torch.save(embeddings, embeddings_file)

        self.enrol_embeddings = {f:embeddings[f] for f in enrol_files}
        self.test_embeddings = {f:embeddings[f] for f in test_files}

    def _extract_train_embeddings(self):
        """
        Extract train embeddings for score normalization.

        Returns:
            None
        """
        # Load training df
        df = pd.read_csv(self.config.dataset.base_path / self.config.dataset.train)
        if "Set" in df.columns:
            df = df[df["Set"] == "train"]

        # Extract or load train embeddings
        embeddings_file = self.config.model_path / f"train_embeddings.pt"
        if embeddings_file.exists():
            self.train_embeddings = torch.load(embeddings_file)
        else:
            files = df["File"].tolist()
            self.train_embeddings = self._extract_embeddings(
                files,
                desc="Extracting train embeddings"
            )
            torch.save(self.train_embeddings, embeddings_file)

    def _compute_norm_stats(self, embeddings: Dict[str, torch.Tensor], batch_size: int = 5000):
        """
        Compute the score normalization statistics.

        Args:
            embeddings (Dict[str, torch.Tensor]): Dict of embeddings tensors.
            batch_size (int): Batch size for internal computations.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Mean and std of scores
        """
        cohort_size = len(self.train_embeddings)
        if self.task_config.score_norm == ScoreNormEnum.ASNORM:
            cohort_size = self.task_config.score_norm_cohort_size

        keys = list(embeddings.keys())

        cohort = torch.stack(list(self.train_embeddings.values())).mean(dim=1) 
        embeddings = torch.stack(list(embeddings.values())).mean(dim=1)

        means = []
        stds = []

        for batch in tqdm(
            embeddings.split(batch_size, dim=0),
            desc='Computing norm stats',
            disable=not self.verbose or not is_main_process()
        ):
            scores = batch @ cohort.T
            scores = torch.topk(scores, k=cohort_size, dim=1).values
            means.append(scores.mean(dim=1))
            stds.append(scores.std(dim=1))

        mean = torch.cat(means, dim=0)
        std = torch.cat(stds, dim=0)

        mean = dict(zip(keys, mean))
        std = dict(zip(keys, std))

        return mean, std

    def _prepare_evaluation(self):
        """
        Prepare evaluation by extracting train and test embeddings.

        Returns:
            None
        """
        self._extract_trials_embeddings(self.task_config.trials)

        if self.task_config.score_norm.value:
            self._extract_train_embeddings()
            self.enrol_mean, self.enrol_std = self._compute_norm_stats(self.enrol_embeddings)
            self.test_mean, self.test_std = self._compute_norm_stats(self.test_embeddings)

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

    def _normalize_score(self, enrol: str, test: str, score: torch.Tensor) -> torch.Tensor:
        """
        Normalize the score.

        Args:
            enrol (str): Enrollment ID.
            test (str): Test ID.
            score (torch.Tensor): Input score tensor.

        Returns:
            torch.Tensor: Output score tensor.
        """
        if self.task_config.score_norm == ScoreNormEnum.ZNORM:
            score = (score - self.enrol_mean[enrol]) / self.enrol_std[enrol]
        elif self.task_config.score_norm == ScoreNormEnum.TNORM:
            score = (score - self.test_mean[test]) / self.test_std[test]
        elif self.task_config.score_norm in (ScoreNormEnum.SNORM, ScoreNormEnum.ASNORM):
            score_e = (score - self.enrol_mean[enrol]) / self.enrol_std[enrol]
            score_t = (score - self.test_mean[test]) / self.test_std[test]
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
        enrol_emb = self.enrol_embeddings[enrol]
        test_emb = self.test_embeddings[test]

        score = self._compute_score(enrol_emb, test_emb)

        score = self._normalize_score(enrol, test, score)

        return score.item()
