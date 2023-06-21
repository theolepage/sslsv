from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from sslsv.evaluation._ClassifierEvaluation import (
    ClassifierEvaluation,
    ClassifierEvaluationTaskConfig
)


@dataclass
class LogisticRegressionEvaluationTaskConfig(ClassifierEvaluationTaskConfig):

    pass


class LogisticRegressionEvaluation(ClassifierEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_classifier(self):
        return LogisticRegression(random_state=0)