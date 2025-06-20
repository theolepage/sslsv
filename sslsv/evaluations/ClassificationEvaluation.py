from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from sslsv.evaluations._BaseEvaluation import BaseEvaluation, EvaluationTaskConfig


@dataclass
class ClassificationEvaluationTaskConfig(EvaluationTaskConfig):
    """
    Classification evaluation configuration.

    Attributes:
        csv (str): Path to the evaluation .csv file.
        key (str): Key of row to extract labels from evaluation .csv file.
    """

    csv: str = None
    key: str = None


class ClassificationEvaluation(BaseEvaluation):
    """
    Classification evaluation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Classification evaluation.

        Args:
            *args: Positional arguments for base class.
            **kwargs: Keyword arguments for base class.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

    def _extract_embeddings_inference(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to perform model inference with classifier and outputs a class id.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        Y = self.model(X, training=True)
        Y = F.softmax(Y, dim=-1)
        Y = torch.argmax(Y, dim=-1)
        return Y

    def _get_embeddings(self, file: Path, subset: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from a .csv file.

        Args:
            file (Path): Path to the evaluation .csv file.
            subset (str): Subset of the data to extract embeddings for.

        Returns:
            Tuple(np.ndarray, np.ndarray): Embeddings (X) and their corresponding labels (y).

        Raises:
            AssertionError: If the number of frames is not equal to 1.
        """
        assert self.task_config.num_frames == 1

        df = pd.read_csv(self.config.dataset.base_path / file)
        df["Label"] = pd.factorize(df[self.task_config.key])[0]
        if "Set" in df.columns:
            df = df[df["Set"] == subset]

        X = self._extract_embeddings(
            df["File"].tolist(), desc=f"Extracting {subset} embeddings"
        )
        X = torch.stack(list(X.values())).numpy().squeeze()

        y = np.array(df["Label"])

        return X, y

    def _get_metrics(
        self,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        file: Path,
    ) -> Dict[str, float]:
        """
        Determine metrics.

        Args:
            y_test (np.ndarray): True labels.
            y_test_pred (np.ndarray): Predicted labels.
            file (Path): Path to the evaluation .csv file.

        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        prefix = file[:-4]

        accuracy = accuracy_score(y_test, y_test_pred)
        f1score = f1_score(y_test, y_test_pred, average="weighted")

        metrics = {
            f"{prefix}/accuracy": accuracy,
            f"{prefix}/f1_score": f1score,
        }

        return metrics

    def _evaluate_file(self, file: Path) -> Dict[str, float]:
        """
        Evaluate on an evaluation .csv file and return embeddings.

        Args:
            file (Path): Path to the evaluation .csv file.

        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        y_test_pred, y_test = self._get_embeddings(file, "test")

        return self._get_metrics(y_test, y_test_pred, file)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on classification.

        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        metrics = {}
        for file in [self.task_config.csv]:
            metrics.update(self._evaluate_file(file))

        return metrics
