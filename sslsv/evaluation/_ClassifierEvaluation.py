from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from sslsv.evaluation._BaseEvaluation import (
    BaseEvaluation,
    EvaluationTaskConfig
)


@dataclass
class ClassifierEvaluationTaskConfig(EvaluationTaskConfig):

    csv: str = None
    key: str = None


class ClassifierEvaluation(BaseEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_embeddings(self, file, subset):
        assert (
            self.config.evaluation.num_frames == 1 or
            self.config.evaluation.mean_of_features
        )

        df = pd.read_csv(self.config.data.base_path / file)
        df = df[df['Set'] == subset]

        X = self._extract_embeddings(
            df['File'].tolist()[:50],
            numpy=True,
            desc=f'Extracting {subset} embeddings'
        )
        X = np.array(list(X.values())).squeeze(axis=1)
        
        y = df[self.task_config.key].tolist()[:50]

        return X, y

    def _get_metrics(self, y_test, y_test_pred, file):
        prefix = file[:-4]

        accuracy = accuracy_score(y_test, y_test_pred)
        f1score = f1_score(y_test, y_test_pred, average='macro')

        metrics = {
            f'{prefix}/accuracy': accuracy,
            f'{prefix}/f1_score': f1score
        }

        return metrics

    def _prepare_evaluation(self):
        X_train, y_train = self._get_embeddings(self.task_config.csv, 'train')

        self.classifier = self._get_classifier()
        self.classifier.fit(X_train, y_train)

    def _evaluate_file(self, file):
        X_test, y_test = self._get_embeddings(file, 'test')

        y_test_pred = self.classifier.predict(X_test)

        return self._get_metrics(y_test, y_test_pred, file)

    def evaluate(self):
        self._prepare_evaluation()

        metrics = {}
        for file in [self.task_config.csv]:
            metrics.update(self._evaluate_file(file))

        return metrics