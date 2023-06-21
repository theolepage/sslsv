from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from sslsv.evaluation._ClassifierEvaluation import (
    ClassifierEvaluation,
    ClassifierEvaluationTaskConfig
)


@dataclass
class LinearRegressionEvaluationTaskConfig(ClassifierEvaluationTaskConfig):

    pass


class LinearRegressionEvaluation(ClassifierEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_metrics(self, y_test, y_test_pred, file):
        prefix = file[:-4]

        mae = mean_absolute_error(y_test, y_test_pred)

        metrics = {
            f'{prefix}/mae': mae
        }

        return metrics

    def _get_classifier(self):
        return LinearRegression()