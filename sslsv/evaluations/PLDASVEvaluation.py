from dataclasses import dataclass
from typing import List, Optional

from pathlib import Path
import numpy as np
import pickle
import pandas as pd

from speechbrain.processing.PLDA_LDA import (
    PLDA,
    StatObject_SB,
    Ndx,
    fast_PLDA_scoring,
)

from sslsv.evaluations._SpeakerVerificationEvaluation import (
    SpeakerVerificationEvaluation,
    SpeakerVerificationEvaluationTaskConfig,
)


def create_stat_object(
    modelset: np.ndarray,
    segset: Optional[np.ndarray],
    embeddings: np.ndarray,
) -> StatObject_SB:
    """
    Create a StatObject_SB object for SpeechBrain.

    Args:
        modelset (np.ndarray): An array of model data.
        segset (Optional[np.ndarray]): An optional array of segmentation data.
        embeddings (np.ndarray): An array of embeddings data.

    Returns:
        StatObject_SB: StatObject_SB object.
    """
    return StatObject_SB(
        modelset=modelset,
        segset=segset,
        start=None,
        stop=None,
        stat0=None,
        stat1=embeddings,
    )


@dataclass
class PLDASVEvaluationTaskConfig(SpeakerVerificationEvaluationTaskConfig):
    """
    PLDA-based Speaker Verification (SV) evaluation configuration.
    """

    pass


class PLDASVEvaluation(SpeakerVerificationEvaluation):
    """
    PLDA-based (PLDA back-end) Speaker Verification (SV) evaluation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a PLDA-based SV evaluation.

        Args:
            *args: Positional arguments for base class.
            **kwargs: Keyword arguments for base class.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

    def _prepare_evaluation_aux(self, trials: List[Path], key: str) -> StatObject_SB:
        """
        Prepare evaluation by extracting embeddings and create a StatObject_SB object.

        Args:
            trials (List[Path]): List of trials file paths.
            key (str): Whether to prepare evaluation data for training, enrollment or test.

        Returns:
            StatObject_SB: StatObject_SB object.
        """
        stat_path = self.config.model_path / f"plda_{key}_stat.pkl"

        if stat_path.exists():
            with open(stat_path, "rb") as f:
                stat = pickle.load(f)
            return stat

        if key == "train":
            df = pd.read_csv(self.config.dataset.base_path / self.config.dataset.train)
            files = df["File"].tolist()
            labels = pd.factorize(df["Speaker"])[0].tolist()
        else:
            files = list(
                dict.fromkeys(
                    [
                        line.rstrip().split()[1 if key == "enrolment" else 2]
                        for trial_file in trials
                        for line in open(self.config.dataset.base_path / trial_file)
                    ]
                )
            )
            labels = None

        embeddings = self._extract_embeddings(
            files, labels, desc=f"Extracting {key} embeddings", numpy=True
        )

        # Convert embeddings from dict to numpy arrays
        embeddings_keys = np.array(list(embeddings.keys()))
        embeddings_values = np.array(list(embeddings.values())).squeeze(axis=1)

        assert self.task_config.num_frames == 1

        if key == "train":
            stat = create_stat_object(np.array(labels), None, embeddings_values)
        else:
            stat = create_stat_object(
                embeddings_keys, embeddings_keys, embeddings_values
            )

        stat.save_stat_object(stat_path)

        return stat

    def _prepare_evaluation(self):
        """
        Prepare evaluation by training the PLDA backend and scoring trials.

        Returns:
            None
        """
        trials = self.task_config.trials

        train_stat = self._prepare_evaluation_aux(trials, "train")

        plda = PLDA()
        plda.plda(train_stat)

        enrolment_stat = self._prepare_evaluation_aux(trials, "enrolment")
        test_stat = self._prepare_evaluation_aux(trials, "test")

        ndx = Ndx(models=enrolment_stat.modelset, testsegs=test_stat.segset)

        self.scores = fast_PLDA_scoring(
            enrolment_stat,
            test_stat,
            ndx,
            plda.mean,
            plda.F,
            plda.Sigma,
        )

    def _get_sv_score(self, enrol: str, test: str) -> float:
        """
        Get the score for a given enrollment and test trial.

        Args:
            enrol (str): Enrollment ID.
            test (str): Test ID.

        Returns:
            float: Score value.
        """
        score = self.scores.scoremat[
            self.scores.modelset == enrol, self.scores.segset == test
        ]
        return score.item()
