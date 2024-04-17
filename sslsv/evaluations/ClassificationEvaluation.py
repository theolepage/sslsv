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

    csv: str = None
    key: str = None


class ClassificationEvaluation(BaseEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_embeddings_inference(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.model(X, training=True)
        Y = F.softmax(Y, dim=-1)
        Y = torch.argmax(Y, dim=-1)
        return Y

    def _get_embeddings(self, file: Path, subset: str) -> Tuple[np.ndarray, np.ndarray]:
        assert self.config.evaluation.num_frames == 1

        df = pd.read_csv(self.config.dataset.base_path / file)
        df["Label"] = pd.factorize(df[self.task_config.key])[0]
        df = df[df["Set"] == subset]

        X = self._extract_embeddings(
            df["File"].tolist(), numpy=True, desc=f"Extracting {subset} embeddings"
        )
        X = np.array(list(X.values())).squeeze()

        y = np.array(df["Label"])

        return X, y

    def _get_metrics(
        self,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        file: Path,
    ) -> Dict[str, float]:
        prefix = file[:-4]

        accuracy = accuracy_score(y_test, y_test_pred)
        f1score = f1_score(y_test, y_test_pred, average="macro")

        metrics = {
            f"{prefix}/accuracy": accuracy,
            f"{prefix}/f1_score": f1score,
        }

        return metrics

    def _evaluate_file(self, file: Path) -> Dict[str, float]:
        y_test_pred, y_test = self._get_embeddings(file, "test")

        return self._get_metrics(y_test, y_test_pred, file)

    def evaluate(self) -> Dict[str, float]:
        metrics = {}
        for file in [self.task_config.csv]:
            metrics.update(self._evaluate_file(file))

        return metrics
