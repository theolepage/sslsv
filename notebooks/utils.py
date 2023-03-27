import math
import pickle5 as pickle

import numpy as np
import torch
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from plotnine import *
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from sslsv.utils.helpers import load_config, load_model
from sslsv.utils.evaluate import CosineEvaluation
from sslsv.utils.evaluate import compute_eer
from sslsv.utils.evaluate import compute_min_dcf


def load_models(MODELS_TO_LOAD, OVERRIDE_MODELS_NAMES):
    MODELS = {}

    for model in MODELS_TO_LOAD:
        config, checkpoint_dir = load_config(model, verbose=False)

        with open(checkpoint_dir + '/embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        model_name = model.split('/')[-1][:-4]
        if model_name in OVERRIDE_MODELS_NAMES.keys():
            model_name = OVERRIDE_MODELS_NAMES[model_name]

        MODELS[model_name] = {
            'config': config,
            'checkpoint_dir': checkpoint_dir,
            'embeddings': embeddings
        }
        
    return MODELS


def compute_scores(MODELS):
    for model_name, model in MODELS.items():
        evaluation = CosineEvaluation(
            model=None,
            config=model['config'],
            device='cpu',
            verbose=False
        )

        evaluation.test_embeddings = model['embeddings']
        
        trial_file = model['config'].data.base_path / model['config'].data.val
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
            p_target=model['config'].evaluate.mindcf_p_target,
            c_miss=model['config'].evaluate.mindcf_c_miss,
            c_fa=model['config'].evaluate.mindcf_c_fa
        )
        
        MODELS[model_name].update({
            'scores': scores,
            'labels': labels,
            'fprs': fprs,
            'fnrs': fnrs,
            'eer': eer,
            'mindcf': mindcf
        })


def show_metrics(MODELS):
    metrics = pd.DataFrame()

    for model_name, model in MODELS.items():
        metrics = metrics.append({
            'Model': model_name,
            'EER(%)': round(model['eer'], 2),
            'minDCF': round(model['mindcf'], 4)
        }, ignore_index=True)
        
    metrics = metrics.set_index('Model')
    metrics = metrics.sort_values(by=['EER(%)'], ascending=False)
    return metrics


def ppndf(p):
    SPLIT = 0.42
    A0 = 2.5066282388
    A1 = -18.6150006252
    A2 = 41.3911977353
    A3 = -25.4410604963
    B1 = -8.4735109309
    B2 = 23.0833674374
    B3 = -21.0622410182
    B4 = 3.1308290983
    C0 = -2.7871893113
    C1 = -2.2979647913
    C2 = 4.8501412713
    C3 = 2.3212127685
    D1 = 3.5438892476
    D2 = 1.6370678189
    LL = 140
    eps = 2.2204e-16

    if p >= 1.0:
        p = 1 - eps
    if p <= 0.0:
        p = eps

    q = p - 0.5
    if abs(q) <= SPLIT:
        r = q * q
        retval = q * (((A3 * r + A2) * r + A1) * r + A0) / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0)
    else:
        r = (1.0 - p) if q > 0.0 else p
        if r <= 0.0:
            print("Found r = ", r)
        r = math.sqrt(-1.0 * math.log(r))
        retval = (((C3 * r + C2) * r + C1) * r + C0) / ((D2 * r + D1) * r + 1.0)
        if q < 0:
            retval *= -1.0
    return round(retval, 5)


def determine_det_curve_pts(scores, labels, p_target=0.01, c_miss=10, c_fa=1):
    pos = [s for i, s in enumerate(scores) if labels[i] == 1]
    neg = [s for i, s in enumerate(scores) if labels[i] == 0]
    pts_x = []
    pts_y = []

    pos.sort()
    neg.sort()

    pts_x.append(ppndf(0))
    pts_y.append(ppndf(1))

    ntrue = 0
    nfalse = 0

    eer_idx = 0
    mindcf_idx = 0
    mindcf = float('inf')
    mindcf_coords = None
    i = 0

    while (ntrue < len(pos)) or (nfalse < len(neg)):
        if nfalse >= len(neg) or (ntrue < len(pos) and pos[ntrue] <= neg[nfalse]):
            ntrue += 1
        else:
            nfalse += 1
        
        i += 1

        p_miss = ntrue / len(pos)
        p_fa = (len(neg) - nfalse) / len(neg)

        x = ppndf(p_miss)
        y = ppndf(p_fa)

        if p_miss <= p_fa:
            eer_idx = i

        dcf = (c_miss * p_miss * p_target) + (c_fa * p_fa * (1 - p_target))
        if dcf <= mindcf:
            mindcf = dcf
            mindcf_idx = i
            mindcf_coords = (x, y)

        pts_x.append(x)
        pts_y.append(y)
    
    return (pts_x, pts_y), eer_idx, mindcf_coords


def show_det_curve(MODELS):
    scale_labels = ['0.001', '0.01', '0.1', '0.5', '1', '2', '5', '10', '20', '30', '40', '50', '80']
    scale_breaks = [-4.26, -3.72, -3.08, -2.57, -2.32635, -2.05, -1.64, -1.28, -0.84, -0.52, -0.25, 0.0, 0.84]

    plot = (
        ggplot()
        + xlab('False Positive Rate (%)')
        + ylab('False Negative Rate (%)')
        + ggtitle('Detection Error Tradeoff (DET) Curve')
        + theme_bw()
        + theme(figure_size=(6, 6), text=element_text(size=10))
        + scale_x_continuous(
            breaks=scale_breaks,
            labels=scale_labels,
        )
        + scale_y_continuous(
            breaks=scale_breaks,
            labels=scale_labels,
        )
        + coord_cartesian(xlim=(-4.40, 0.0), ylim=(-3.29052, 1.0), expand=False)
    )

    labels = []

    for model_name, model in MODELS.items():
        pts, eer_idx, mindcf_coords = determine_det_curve_pts(
            model['scores'],
            model['labels']
        )

        df = pd.DataFrame()
        df['fprs'] = pts[1]
        df['fnrs'] = pts[0]
        df['model'] = model_name
        
        df = df.sort_values(by=['fprs', 'fnrs'], ascending=[True, False])
        
        plot += geom_line(
            df,
            aes(x='fprs', y='fnrs', color='model'),
            size=0.5
        )
        plot += geom_point(
            df,
            aes(
                x=pts[0][eer_idx],
                y=pts[1][eer_idx],
                color='model',
                shape="'1'"
            ),
            size=2
        )
        plot += geom_point(
            df,
            aes(
                x=mindcf_coords[1],
                y=mindcf_coords[0],
                color='model',
                shape="'2'"
            ),
            size=2
        )
        
        label = model_name 
    #     label += f" (EER: {round(model['eer'], 2)}%"
    #     label += f", minDCF: {round(model['min_dcf'], 4)})"
        labels.append(label)

    plot += geom_abline(size=0.05)
        
    plot += scale_color_discrete(name='Models', labels=labels)
    plot += scale_shape_manual(name='Metrics', labels=('EER', 'minDCF'), values=('o', '^'))

    return plot


def show_scores_distribution(MODELS, use_angle=False):
    df = []
    means = []
    for model_name, model in MODELS.items():
        df_ = pd.DataFrame()
        df_['Score'] = [
            (s if not use_angle else torch.acos(s).item())
            for s in model['scores']
        ]
        df_['Target'] = [
            ('Positive' if l else 'Negative')
            for l in model['labels']]
        df_['Model'] = model_name
        df.append(df_)
        
        neg_mean = df_[df_['Target'] == 'Negative'].Score.mean()
        pos_mean = df_[df_['Target'] == 'Positive'].Score.mean()
        neg_std = df_[df_['Target'] == 'Negative'].Score.std()
        pos_std = df_[df_['Target'] == 'Positive'].Score.std()
        print(f'{model_name} Diff mean: {pos_mean-neg_mean}')
        means.append((neg_mean, pos_mean))
        
    df = pd.concat(df)

    df['Model'] = df['Model'].astype('category')
    df['Model'] = df['Model'].cat.reorder_categories(MODELS.keys())

    plot = (
        ggplot()
        + xlab('Score' if not use_angle else 'Angle')
        + ylab('Count')
        + ggtitle(f'Scores distribution')
        + theme_bw()
        + theme(figure_size=(10, 4), text=element_text(size=10))
        + geom_histogram(
            df,
            aes(x='Score', fill='Target', color='Target'),
            alpha=0.6,
            binwidth=0.012,
            position='identity'
        )
        + facet_wrap('Model')
    )

    for i, (model_name, model) in enumerate(MODELS.items()):
        plot += geom_vline(df[df['Model'] == model_name], aes(xintercept=means[i][0]), size=0.3, linetype='dashed')
        plot += geom_vline(df[df['Model'] == model_name], aes(xintercept=means[i][1]), size=0.3, linetype='dashed')

    return plot


def filter_embeddings(embeddings, nb_speakers, nb_samples):
    # Determine list of speakers id
    speakers_id = [key[:-4].split('/')[0] for key in embeddings.keys()]
    speakers_id = [
        s
        for s in list(set(speakers_id))
        if speakers_id.count(s) >= nb_samples
    ]
    speakers_id = speakers_id[:nb_speakers]
    
    Z, y = [], []
    for s, speaker_id in enumerate(speakers_id):
        i = 0
        for key in embeddings.keys():
            if i < nb_samples and speaker_id == key[:-4].split('/')[0]:
                Z.append(embeddings[key].numpy()[0])
                y.append(s)
                i += 1
    
    return np.array(Z), y


def plot_3d_tsne(model, nb_speakers=7, nb_samples=150):
    Z, y = filter_embeddings(model['embeddings'], nb_speakers, nb_samples)

    Z = TSNE(n_components=3, init='random').fit_transform(Z)

    # Project on unit sphere
    Z = Z / np.expand_dims(np.sqrt(Z[:, 0] ** 2 + Z[:, 1] ** 2 + Z[:, 2] ** 2), -1)
    
    def add_unit_sphere(resolution=50):
        u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
        X_s = np.cos(u)*np.sin(v)
        Y_s = np.sin(u)*np.sin(v)
        Z_s = np.cos(v)

        return go.Surface(
            x=X_s,
            y=Y_s,
            z=Z_s,
            opacity=0.1,
            colorscale='Greys',
            showscale=False
        )

    data = [
    #     add_unit_sphere(),
        go.Scatter3d(
            x=Z[:, 0], 
            y=Z[:, 1], 
            z=Z[:, 2], 
            opacity=0.8,
            mode='markers',
            marker=dict(
                size=3,
                color=y,
                colorscale='Viridis',
                opacity=0.8
            )
        )
    ]

    fig = go.Figure(data=data)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


def plot_2d_tsne(model, nb_speakers=10, nb_samples=150):
    Z, y = filter_embeddings(model['embeddings'], nb_speakers, nb_samples)

    Z_2d = TSNE(n_components=2, init='random').fit_transform(Z)

    df = pd.DataFrame(Z)
    df['Speaker'] = y
    df['t-SNE_1'] = Z_2d[:, 0]
    df['t-SNE_2'] = Z_2d[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x='t-SNE_1',
        y='t-SNE_2',
        hue='Speaker',
        palette=sns.color_palette('hls', len(np.unique(y))),
        data=df,
        legend='full',
        alpha=0.5
    )
    plt.show()