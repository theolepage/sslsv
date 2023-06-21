from dataclasses import dataclass, field
from typing import List

import numpy as np

from sslsv.evaluation._BaseEvaluation import (
    BaseEvaluation,
    EvaluationTaskConfig
)

from sslsv.evaluation.sv_metrics import (
    compute_eer,
    compute_mindcf,
    compute_actdcf,
    compute_cllr,
    compute_avgrprec
)


def compute_error_rates(scores, targets):
    scores = np.array(scores)
    targets = np.array(targets)

    nb_target_scores = len(scores[targets == 1])
    nb_nontarget_scores = len(scores[targets == 0])

    sorted_idx = np.argsort(scores)

    # Determine the number of positives that will be classified
    # as negatives as the score/threshold increases. 
    sum_fn = np.cumsum(targets[sorted_idx])
    
    # Determine the number of negatives that will be classified
    # as positives as the score/threshold increases. 
    sum_fp = np.cumsum(np.where(targets[sorted_idx] == 0, 1, 0))

    fnrs = np.empty(len(scores) + 1)
    fnrs[0] = 0
    fnrs[1:] = sum_fn / nb_target_scores
    
    fprs = np.empty(len(scores) + 1)
    fprs[0] = 1
    fprs[1:] = (nb_nontarget_scores - sum_fp) / nb_nontarget_scores

    return fprs, fnrs, scores[sorted_idx]


@dataclass
class SpeakerVerificationEvaluationTaskConfig(EvaluationTaskConfig):

    trials: List[str] = field(default_factory=lambda: [
        'voxceleb1_test_O',
        # 'voxceleb1_test_H',
        # 'voxceleb1_test_E',
        # 'voxsrc2021_val',
        # 'voices2019_dev'
    ])

    metrics: List[str] = field(default_factory=lambda: [
        'eer',
        'mindcf',
        # 'actdcf',
        # 'cllr',
        # 'avgrprec'
    ])

    mindcf_p_target: float = 0.01
    mindcf_c_miss: float = 1
    mindcf_c_fa: float = 1


class SpeakerVerificationEvaluation(BaseEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_sv_score(self, a, b):
        raise NotImplementedError

    def _get_metrics(self, trials, scores, targets, file):
        metrics = {}

        fprs, fnrs, sorted_scores = compute_error_rates(scores, targets)

        if 'eer' in self.task_config.metrics:
            metrics[f'{file}/eer'] = compute_eer(fprs, fnrs)

        if 'mindcf' in self.task_config.metrics:
            metrics[f'{file}/mindcf'] = compute_mindcf(
                fprs,
                fnrs,
                p_target=self.task_config.mindcf_p_target,
                c_miss=self.task_config.mindcf_c_miss,
                c_fa=self.task_config.mindcf_c_fa
            )

        if 'actdcf' in self.task_config.metrics:
            metrics[f'{file}/actdcf'] = compute_actdcf(
                fprs,
                fnrs,
                sorted_scores,
                p_target=self.task_config.mindcf_p_target,
                c_miss=self.task_config.mindcf_c_miss,
                c_fa=self.task_config.mindcf_c_fa
            )

        if 'cllr' in self.task_config.metrics:
            metrics[f'{file}/cllr'] = compute_cllr(scores, targets)

        if 'avgrprec' in self.task_config.metrics:
            metrics[f'{file}/avgrprec'] = compute_avgrprec(trials)
        
        return metrics

    def _evaluate_trials(self, file):
        trials, scores, targets = {}, [], []

        with open(self.config.data.base_path / file) as f:
            lines = f.readlines()
        for line in lines: #tqdm(lines, desc='Computing scores')
            target, enrol, test = line.rstrip().split(' ')

            score = self._get_sv_score(enrol, test)
            target = int(target)

            scores.append(score)
            targets.append(target)

            if enrol not in trials: trials[enrol] = ([], [])
            trials[enrol][0].append(target)
            trials[enrol][1].append(score)

        self.scores = scores
        self.targets = targets

        return self._get_metrics(trials, scores, targets, file)

    def evaluate(self):
        self._prepare_evaluation()

        metrics = {}
        for file in self.task_config.trials:
            metrics.update(self._evaluate_trials(file))

        return metrics