import torch
import torch.nn.functional as F

import numpy as np
import soundfile as sf

from sklearn.metrics import roc_curve

from sslsv.data.utils import load_audio


def extract_embeddings__(curr_batch_data, model, config):
    batch = torch.stack(curr_batch_data, dim=0)
    B, N, T = batch.shape
    batch = batch.reshape((B * N, T))

    with torch.no_grad():
        batch = batch.cuda() if torch.cuda.is_available() else batch
        feats = model(batch).detach().cpu()

    feats = feats.reshape((B, N, -1))
    if config.evaluate.mean_of_features:
        feats = feats.mean(axis=1, keepdim=True)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats


def extract_embeddings_(
    model,
    trials,
    config,
    frame_length,
    batch_size,
    num_frames
):
    # Get a list of unique utterances
    utterances = set()
    for trial_file in trials:
        for line in open(config.data.base_path / trial_file):
            target, a, b = line.rstrip().split(' ')
            utterances.add(a)
            utterances.add(b)

    # Determine embeddings for each unique utterance
    embeddings = {}
    curr_batch_ids = []
    curr_batch_data = []
    for utterance in utterances:
        if len(curr_batch_ids) == batch_size:
            feats = extract_embeddings__(curr_batch_data, model, config)
            for i in range(len(curr_batch_ids)):
                uttid, data = curr_batch_ids[i], feats[i]
                embeddings[uttid] = data
            curr_batch_ids, curr_batch_data = [], []

        # Store current utterance id and data
        audio_path = config.data.base_path / 'voxceleb1' / utterance
        data = load_audio(audio_path, frame_length, num_frames)
        curr_batch_ids.append(utterance)
        curr_batch_data.append(torch.FloatTensor(data))

    # Register remaining samples (if nb samples % batch_size != 0)
    if len(curr_batch_ids) != 0:
        feats = extract_embeddings__(curr_batch_data, model, config)
        for i in range(len(curr_batch_ids)):
            uttid, data = curr_batch_ids[i], feats[i]
            embeddings[uttid] = data

    return embeddings


def extract_embeddings(model, trials, config):
    embeddings = []
    embeddings.append(
        extract_embeddings_(
            model,
            trials,
            config,
            frame_length=config.evaluate.frame_length,
            batch_size=config.evaluate.batch_size,
            num_frames=config.evaluate.num_frames
        )
    )
    if config.evaluate.average_with_full_length:
        embeddings.append(
            extract_embeddings_(
                model,
                trials,
                config,
                frame_length=None,
                batch_size=1,
                num_frames=1
            )
        )
    return embeddings

def score_trials(trials_path, embeddings):
    scores, labels = [], []
    for line in open(trials_path):
        target, a, b = line.rstrip().split(' ')

        score = 0
        for embeddings_ in embeddings:
            score += torch.mean(embeddings_[a] @ embeddings_[b].T)
        score /= len(embeddings)
        label = int(target)

        scores.append(score)
        labels.append(label)

    return scores, labels


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


def evaluate(model, config, validation=False):
    trials = [config.data.val] if validation else config.data.test

    embeddings = extract_embeddings(model, trials, config)

    # Determine metrics
    metrics = {}
    for trial_file in trials:
        scores, labels = score_trials(
            config.data.base_path / trial_file,
            embeddings
        )

        fprs, tprs, thresholds = roc_curve(
            labels,
            scores,
            pos_label=1,
            drop_intermediate=False
        )
        fnrs = 1 - tprs

        eer, _ = compute_eer(fprs, fnrs)
        mindcf, _ = compute_min_dcf(fprs, fnrs, p_target=0.01)

        key = 'val' if validation else f'test/{trial_file}'
        metrics = {
            **metrics,
            f'{key}/eer': eer,
            f'{key}/mindcf': mindcf
        }

    return metrics, embeddings