import os
from operator import itemgetter

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
    config,
    frame_length,
    batch_size,
    num_frames
):
    # Get a list of unique utterances
    utterances = set()
    for line in open(config.data.trials):
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
        audio_path = os.path.join(config.data.base_path, 'voxceleb1', utterance)
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


def extract_embeddings(model, config):
    embeddings = []
    embeddings.append(
        extract_embeddings_(
            model,
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


def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idxE = np.nanargmin(np.abs(fnr - fpr))
    eer  = max(fpr[idxE], fnr[idxE]) * 100
    return eer


def compute_error_rates(scores, labels):
      # Sort scores from smallest to largest.
      # Scores are the thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      labels = [labels[i] for i in sorted_indexes]

      # Determine false negative rates and false positive rates for each threshold.
      fnrs = []
      fprs = []
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i -1] + labels[i])
              fprs.append(fprs[i -1] + 1 - labels[i])

      fnrs_norm = sum(labels)
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      fprs_norm = len(labels) - fnrs_norm
      fprs = [1 - x / float(fprs_norm) for x in fprs]

      return fnrs, fprs


def compute_min_dcf(fnrs, fprs, p_target=0.01, c_miss=1, c_fa=1):
    # Equations are from Section 3 of
    # NIST 2016 Speaker Recognition Evaluation Plan

    # Equation (2)
    min_c_det = float('inf')
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det

    # Equations (3) and (4)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf


def evaluate(embeddings, trials):
    scores, labels = score_trials(trials, embeddings)

    eer = compute_eer(scores, labels)

    fnrs, fprs = compute_error_rates(scores, labels)
    min_dcf_001 = compute_min_dcf(fnrs, fprs, p_target=0.01)

    return { 'test_eer': eer, 'test_min_dcf_001': min_dcf_001 }
