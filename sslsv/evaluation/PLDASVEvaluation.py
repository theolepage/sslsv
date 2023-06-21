from dataclasses import dataclass

import numpy as np
import pickle
from pathlib import Path
import pandas as pd

from speechbrain.processing.PLDA_LDA import (
    PLDA,
    StatObject_SB,
    Ndx,
    fast_PLDA_scoring
)

from sslsv.evaluation._SpeakerVerificationEvaluation import (
    SpeakerVerificationEvaluation,
    SpeakerVerificationEvaluationTaskConfig
)


@dataclass
class PLDASVEvaluationTaskConfig(SpeakerVerificationEvaluationTaskConfig):

    pass


class PLDASVEvaluation(SpeakerVerificationEvaluation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_evaluation_aux(self, trials, key):
        stat_path = Path(f'./checkpoints/{self.config.name}/plda_{key}_stat.pkl')

        if stat_path.exists():
            with open(stat_path, "rb") as f:
                stat = pickle.load(f)
            return stat

        labels = None

        if key == 'train':
            df = pd.read_csv(self.config.data.base_path / self.config.data.train)
            files = df['File'].tolist()
            labels = pd.factorize(df['Speaker'])[0].tolist()
        else:
            k = 1 if key == 'enrolment' else 2
            files = list(dict.fromkeys([
                line.rstrip().split()[k]
                for trial_file in trials
                for line in open(self.config.data.base_path / trial_file)
            ]))

        embeddings = self._extract_embeddings(
            files,
            labels,
            desc=f'Extracting {key} embeddings',
            numpy=True
        )

        assert (
            self.config.evaluation.num_frames == 1 or
            self.config.evaluation.mean_of_features
        )

        modelset = labels if key == 'train' else list(embeddings.keys())
        segset = list(embeddings.keys())
        embeddings = np.array(list(embeddings.values())).squeeze(axis=1)

        modelset = np.array(modelset, dtype="|O")
        segset = np.array(segset, dtype="|O")
        s = np.array([None] * len(embeddings))
        b = np.array([[1.0]] * len(embeddings))

        stat = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings
        )

        stat.save_stat_object(stat_path)

        return stat

    def _prepare_evaluation(self, trials):
        train_stat = self._prepare_evaluation_aux(trials, 'train')

        plda = PLDA()
        plda.plda(train_stat)

        enrolment_stat = self._prepare_evaluation_aux(trials, 'enrolment')
        test_stat = self._prepare_evaluation_aux(trials, 'test')

        self.test_embeddings = None
        
        ndx = Ndx(
            models=enrolment_stat.modelset,
            testsegs=test_stat.modelset
        )

        self.plda_scores = fast_PLDA_scoring(
            enrolment_stat,
            test_stat,
            ndx,
            plda.mean,
            plda.F,
            plda.Sigma
        )

    def _get_sv_score(self, a, b):
        i = int(np.where(self.plda_scores.modelset == a)[0][0])
        j = int(np.where(self.plda_scores.segset == b)[0][0])
        return self.plda_scores.scoremat[i, j].item()