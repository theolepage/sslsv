import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import soundfile as sf
from tqdm import tqdm

import pickle
from pathlib import Path

from speechbrain.processing.PLDA_LDA import (
    PLDA,
    StatObject_SB,
    Ndx,
    fast_PLDA_scoring
)

from sklearn.metrics import roc_curve

from sslsv.utils.helpers import seed_dataloader_worker, get_checkpoint_dir
from sslsv.data.AudioDataset import AudioDataset
from sslsv.data.utils import load_audio


class BaseEvaluation:

    def __init__(self, model, config, device, verbose):
        self.model = model
        self.config = config
        self.device = device
        self.verbose = verbose

    def _get_score(self, a, b):
        raise NotImplementedError

    def _extract_embeddings(self, files, labels=None, desc=None, numpy=False):
        dataset = AudioDataset(
            base_path=self.config.data.base_path,
            files=files,
            labels=labels,
            frame_length=self.config.evaluate.frame_length,
            num_frames=self.config.evaluate.num_frames
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.evaluate.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            worker_init_fn=seed_dataloader_worker
        )

        embeddings = {}

        dataloader = tqdm(dataloader, desc=desc) if self.verbose else dataloader
        for idx, X, info in dataloader:
            B, N, L = X.size()

            X = X.to(self.device)
            X = X.reshape((B * N, L))

            with torch.no_grad():
                Y = self.model(X)

            Y = Y.reshape((B, N, -1))
            if self.config.evaluate.mean_of_features:
                Y = Y.mean(dim=1, keepdim=True)
            Y = F.normalize(Y, p=2, dim=-1)

            embeddings.update({
                info['files'][i]:(Y[i].cpu().numpy() if numpy else Y[i].cpu())
                for i in range(B)
            })

        return embeddings

    def evaluate(self, trials_path):
        scores, labels = [], []
        
        with open(trials_path) as f:
            trials = f.readlines()
        
        trials = tqdm(trials, desc='Computing scores') if self.verbose else trials
        for trial in trials:
            target, a, b = trial.rstrip().split(' ')

            # FIXME: trials lists do not contain `voxceleb1` path prefix
            a = f'voxceleb1/{a}'
            b = f'voxceleb1/{b}'

            scores.append(self._get_score(a, b))
            labels.append(int(target))

        return scores, labels


class CosineEvaluation(BaseEvaluation):

    def __init__(self, model, config, device, verbose):
        super().__init__(model, config, device, verbose)

    def _extract_train_embeddings(self):
        train_files = [
            line.rstrip().split()[1]
            for line
            in open(self.config.data.base_path / self.config.data.train)
        ]

        self.train_embeddings = torch.stack(list(self._extract_embeddings(
            train_files,
            desc='Extracting train embeddings'
        ).values()))

    def _extract_test_embeddings(self, trials):
        test_files = list(dict.fromkeys([
            f'voxceleb1/{line.rstrip().split()[i]}'
            for trial_file in trials
            for line in open(self.config.data.base_path / trial_file)
            for i in (1, 2)
        ]))

        self.test_embeddings = self._extract_embeddings(
            test_files,
            desc='Extracting test embeddings'
        )

    def prepare(self, trials):
        if self.config.evaluate.score_norm:
            self._extract_train_embeddings()

        self._extract_test_embeddings(trials)

    def _compute_score(self, enrol, test):
        return torch.mean(enrol @ test.T, dim=(-2, -1))

    def _compute_norm_stats(self, enrol, test):
        cohort_size = self.config.evaluate.score_norm_cohort_size

        score_e_c = self._compute_score(self.train_embeddings, enrol)
        score_e_c = torch.topk(score_e_c, k=cohort_size, dim=0)[0]
        self.mean_e_c = torch.mean(score_e_c)
        self.std_e_c = torch.std(score_e_c)

        score_t_c = self._compute_score(self.train_embeddings, test)
        score_t_c = torch.topk(score_t_c, k=cohort_size, dim=0)[0]
        self.mean_t_c = torch.mean(score_t_c)
        self.std_t_c = torch.std(score_t_c)

    def _normalize_score(self, score):
        score_norm = self.config.evaluate.score_norm

        if score_norm == 'z-norm':
            score = (score - self.mean_e_c) / self.std_e_c
        elif score_norm == 't-norm':
            score = (score - self.mean_t_c) / self.std_t_c
        elif score_norm == 's-norm':
            score_e = (score - self.mean_e_c) / self.std_e_c
            score_t = (score - self.mean_t_c) / self.std_t_c
            score = (score_e + score_t) / 2
        
        return score

    def _get_score(self, a, b):
        enrol = self.test_embeddings[a]
        test = self.test_embeddings[b]

        if self.config.evaluate.score_norm:
            self._compute_norm_stats(enrol, test)

        score = self._compute_score(enrol, test)

        if self.config.evaluate.score_norm:
            score = self._normalize_score(score)

        return score.item()


class PLDAEvaluation(BaseEvaluation):

    def __init__(self, model, config, device, verbose):
        super().__init__(model, config, device, verbose)

    def prepare_aux(self, trials, key):
        stat_path = Path(get_checkpoint_dir(self.config)) / f'plda_{key}_stat.pkl'

        if stat_path.exists():
            with open(stat_path, "rb") as f:
                stat = pickle.load(f)
            return stat

        labels = None

        if key == 'train':
            train_path = self.config.data.base_path / self.config.data.train
            files =  [line.rstrip().split()[1] for line in open(train_path)]
            labels = [line.rstrip().split()[0] for line in open(train_path)]
        else:
            k = 1 if key == 'enrolment' else 2
            files = list(dict.fromkeys([
                f'voxceleb1/{line.rstrip().split()[k]}'
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
            self.config.evaluate.num_frames == 1 or
            self.config.evaluate.mean_of_features
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

    def prepare(self, trials):
        train_stat = self.prepare_aux(trials, 'train')

        plda = PLDA()
        plda.plda(train_stat)

        enrolment_stat = self.prepare_aux(trials, 'enrolment')
        test_stat = self.prepare_aux(trials, 'test')

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

    def _get_score(self, a, b):
        i = int(np.where(self.plda_scores.modelset == a)[0][0])
        j = int(np.where(self.plda_scores.segset == b)[0][0])
        return self.plda_scores.scoremat[i, j].item()


def compute_eer(fprs, fnrs):
    idx = np.nanargmin(np.abs(fnrs - fprs))
    eer  = max(fprs[idx], fnrs[idx]) * 100
    return eer, idx


def compute_min_dcf(fprs, fnrs, p_target=0.01, c_miss=1, c_fa=1):
    # Equations are from Section 3 of
    # NIST 2016 Speaker Recognition Evaluation Plan

    # Equation (2)
    min_c_det = float('inf')
    min_c_det_idx = None
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_idx = i

    # Equations (3) and (4)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_idx


def evaluate(model, config, device, validation=False, verbose=True):
    trials = [config.data.val] if validation else config.data.test

    REGISTERED_EVALUATIONS = {
        'cosine': CosineEvaluation,
        'plda':   PLDAEvaluation
    }

    eval_method = 'cosine' if validation else config.evaluate.method
    if eval_method not in REGISTERED_EVALUATIONS.keys():
        raise (
            Exception(f'Evaluation method `{eval_method}` not supported')
        )

    evaluation = REGISTERED_EVALUATIONS[eval_method](
        model,
        config,
        device,
        verbose
    )
    evaluation.prepare(trials)

    metrics = {}
    for trial_file in trials:
        trial_file = config.data.base_path / trial_file
        scores, labels = evaluation.evaluate(trial_file)

        fprs, tprs, thresholds = roc_curve(
            labels,
            scores,
            pos_label=1,
            drop_intermediate=False
        )
        fnrs = 1 - tprs

        eer, _ = compute_eer(fprs, fnrs)

        mindcf, _ = compute_min_dcf(
            fprs,
            fnrs,
            p_target=config.evaluate.mindcf_p_target,
            c_miss=config.evaluate.mindcf_c_miss,
            c_fa=config.evaluate.mindcf_c_fa
        )

        key = 'val' if validation else f'test/{trial_file}'
        metrics = {
            **metrics,
            f'{key}/eer': eer,
            f'{key}/mindcf': mindcf
        }

    return metrics, evaluation.test_embeddings