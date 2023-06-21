from dataclasses import dataclass

from sklearn.svm import SVC

from sslsv.evaluation._ClassifierEvaluation import (
    ClassifierEvaluation,
    ClassifierEvaluationTaskConfig
)


@dataclass
class SVMEvaluationTaskConfig(ClassifierEvaluationTaskConfig):

    pass


class SVMEvaluation(ClassifierEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_classifier(self):
        return SVC(random_state=0)