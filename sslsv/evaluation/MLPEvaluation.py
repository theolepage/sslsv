from dataclasses import dataclass

from sklearn.neural_network import MLPClassifier

from sslsv.evaluation._ClassifierEvaluation import (
    ClassifierEvaluation,
    ClassifierEvaluationTaskConfig
)


@dataclass
class MLPEvaluationTaskConfig(ClassifierEvaluationTaskConfig):

    pass


class MLPEvaluation(ClassifierEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_classifier(self):
        return MLPClassifier(random_state=0)