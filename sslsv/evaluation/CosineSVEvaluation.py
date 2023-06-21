from dataclasses import dataclass

import torch
import pandas as pd

from sslsv.evaluation._SpeakerVerificationEvaluation import (
    SpeakerVerificationEvaluation,
    SpeakerVerificationEvaluationTaskConfig
)


@dataclass
class CosineSVEvaluationTaskConfig(SpeakerVerificationEvaluationTaskConfig):

    score_norm: str = None # 's-norm'
    score_norm_cohort_size: int = 20000


class CosineSVEvaluation(SpeakerVerificationEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_train_embeddings(self):
        df = pd.read_csv(self.config.data.base_path / self.config.data.train)
        files = df['File'].tolist()

        self.train_embeddings = torch.stack(list(self._extract_embeddings(
            files,
            desc='Extracting train embeddings'
        ).values()))

    def _extract_test_embeddings(self, trials):
        test_files = list(dict.fromkeys([
            line.rstrip().split()[i]
            for trial_file in trials
            for line in open(self.config.data.base_path / trial_file)
            for i in (1, 2)
        ]))

        self.test_embeddings = self._extract_embeddings(
            test_files,
            desc='Extracting test embeddings'
        )

    def _prepare_evaluation(self):
        if self.task_config.score_norm:
            self._extract_train_embeddings()

        self._extract_test_embeddings(self.task_config.trials)

    def _compute_score(self, enrol, test):
        return torch.mean(enrol @ test.T, dim=(-2, -1))

    def _compute_norm_stats(self, enrol, test):
        cohort_size = self.task_config.score_norm_cohort_size

        score_e_c = self._compute_score(self.train_embeddings, enrol)
        score_e_c = torch.topk(score_e_c, k=cohort_size, dim=0)[0]
        self.mean_e_c = torch.mean(score_e_c)
        self.std_e_c = torch.std(score_e_c)

        score_t_c = self._compute_score(self.train_embeddings, test)
        score_t_c = torch.topk(score_t_c, k=cohort_size, dim=0)[0]
        self.mean_t_c = torch.mean(score_t_c)
        self.std_t_c = torch.std(score_t_c)

    def _normalize_score(self, score):
        score_norm = self.task_config.score_norm

        if score_norm == 'z-norm':
            score = (score - self.mean_e_c) / self.std_e_c
        elif score_norm == 't-norm':
            score = (score - self.mean_t_c) / self.std_t_c
        elif score_norm == 's-norm':
            score_e = (score - self.mean_e_c) / self.std_e_c
            score_t = (score - self.mean_t_c) / self.std_t_c
            score = (score_e + score_t) / 2
        
        return score

    def _get_sv_score(self, a, b):
        enrol = self.test_embeddings[a]
        test = self.test_embeddings[b]

        if self.task_config.score_norm:
            self._compute_norm_stats(enrol, test)

        score = self._compute_score(enrol, test)

        if self.task_config.score_norm:
            score = self._normalize_score(score)

        return score.item()