from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
import plotnine
from plotnine import *
import plotly
import plotly.graph_objects as go
import seaborn as sns

from notebooks.notebooks_utils import Model


def _ppndf(p: float) -> float:
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
        retval = (
            q
            * (((A3 * r + A2) * r + A1) * r + A0)
            / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0)
        )
    else:
        r = (1.0 - p) if q > 0.0 else p
        if r <= 0.0:
            print("Found r = ", r)
        r = math.sqrt(-1.0 * math.log(r))
        retval = (((C3 * r + C2) * r + C1) * r + C0) / ((D2 * r + D1) * r + 1.0)
        if q < 0:
            retval *= -1.0
    return round(retval, 5)


def _determine_det_curve_pts(
    scores: List[float],
    targets: List[int],
    p_target: float = 0.01,
    c_miss: float = 10,
    c_fa: float = 1,
) -> Tuple[Tuple[float, float], int, Tuple[float, float]]:
    pos = [s for i, s in enumerate(scores) if targets[i] == 1]
    neg = [s for i, s in enumerate(scores) if targets[i] == 0]
    pts_x = []
    pts_y = []

    pos.sort()
    neg.sort()

    pts_x.append(_ppndf(0))
    pts_y.append(_ppndf(1))

    ntrue = 0
    nfalse = 0

    eer_idx = 0
    mindcf_idx = 0
    mindcf = float("inf")
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

        x = _ppndf(p_miss)
        y = _ppndf(p_fa)

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


def det_curve(models: Dict[str, Model]) -> plotnine.ggplot:
    scale_labels = [
        "0.001",
        "0.01",
        "0.1",
        "0.5",
        "1",
        "2",
        "5",
        "10",
        "20",
        "30",
        "40",
        "50",
        "80",
    ]
    scale_breaks = [
        -4.26,
        -3.72,
        -3.08,
        -2.57,
        -2.32635,
        -2.05,
        -1.64,
        -1.28,
        -0.84,
        -0.52,
        -0.25,
        0.0,
        0.84,
    ]

    plot = (
        ggplot()
        + labs(x="False Positive Rate (%)", y="False Negative Rate (%)", color="Models")
        # + ggtitle("Detection Error Tradeoff (DET) Curve")
        + theme_bw()
        + theme(
            figure_size=(9.5, 8),
            text=element_text(size=14),
            # legend_position='top',
            # legend_title=element_blank(),
        )
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

    for model_name, model_entry in models.items():
        pts, eer_idx, mindcf_coords = _determine_det_curve_pts(
            model_entry.scores, model_entry.targets
        )

        df = pd.DataFrame()
        df["fprs"] = pts[1][::10]
        df["fnrs"] = pts[0][::10]
        df["model"] = model_name

        df = df.sort_values(by=["fprs", "fnrs"], ascending=[True, False])

        plot += geom_line(df, aes(x="fprs", y="fnrs", color="model"), size=0.5)
        plot += geom_point(
            df,
            aes(x=pts[0][eer_idx], y=pts[1][eer_idx], color="model", shape="'1'"),
            size=2,
        )
        plot += geom_point(
            df,
            aes(x=mindcf_coords[1], y=mindcf_coords[0], color="model", shape="'2'"),
            size=2,
        )

        labels.append(model_name)

    plot += geom_abline(size=0.05)

    plot += scale_color_discrete(labels=labels)
    plot += scale_shape_manual(
        name="Metrics", labels=("EER", "minDCF"), values=("o", "^")
    )

    return plot


def scores_distribution(
    models: Dict[str, Model],
    use_angle: bool = False,
) -> plotnine.ggplot:
    df = []
    means = []
    for model_name, model_entry in models.items():
        df_ = pd.DataFrame()
        df_["Score"] = [
            (s if not use_angle else torch.acos(s).item()) for s in model_entry.scores
        ]
        df_["Target"] = [("Positive" if l else "Negative") for l in model_entry.targets]
        df_["Model"] = model_name
        df.append(df_)

        neg_mean = df_[df_["Target"] == "Negative"].Score.mean()
        pos_mean = df_[df_["Target"] == "Positive"].Score.mean()
        means.append((neg_mean, pos_mean))

    df = pd.concat(df)

    df["Model"] = df["Model"].astype("category")
    df["Model"] = df["Model"].cat.reorder_categories(models.keys())

    plot = (
        ggplot()
        + xlab("Score" if not use_angle else "Angle")
        + ylab("Count")
        + ggtitle(f"Scores distribution")
        + theme_bw()
        + theme(figure_size=(12, 6), text=element_text(size=10))
        + geom_histogram(
            df,
            aes(x="Score", fill="Target", color="Target"),
            alpha=0.6,
            binwidth=0.012,
            position="identity",
        )
        + facet_wrap("Model")
    )

    for i, (model_name, model) in enumerate(models.items()):
        plot += geom_vline(
            df[df["Model"] == model_name],
            aes(xintercept=means[i][0]),
            size=0.3,
            linetype="dashed",
        )
        plot += geom_vline(
            df[df["Model"] == model_name],
            aes(xintercept=means[i][1]),
            size=0.3,
            linetype="dashed",
        )

    return plot


def _filter_embeddings(
    embeddings: Dict[str, torch.Tensor],
    nb_speakers: int,
    nb_samples: int,
    speakers: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[int]]:
    if speakers is None:
        speakers = [key.split("/")[-3] for key in embeddings.keys()]
        speakers = [s for s in list(set(speakers)) if speakers.count(s) >= nb_samples]
        speakers = speakers[:nb_speakers]

    Z, y = [], []
    for speaker in speakers:
        i = 0
        for key in embeddings.keys():
            if i < nb_samples and speaker == key.split("/")[-3]:
                Z.append(embeddings[key].numpy()[0])
                y.append(speaker)
                i += 1

    return np.array(Z), y


def tsne_3D(
    model: Model,
    nb_speakers: int = 7,
    nb_samples: int = 150,
) -> plotly.graph_objs.Figure:
    Z, y = _filter_embeddings(model.embeddings, nb_speakers, nb_samples)

    Z = TSNE(n_components=3, init="random").fit_transform(Z)

    # Project on unit sphere
    Z = Z / np.expand_dims(np.sqrt(Z[:, 0] ** 2 + Z[:, 1] ** 2 + Z[:, 2] ** 2), -1)

    def add_unit_sphere(resolution: int = 50) -> plotly.graph_objs.Surface:
        u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
        X_s = np.cos(u) * np.sin(v)
        Y_s = np.sin(u) * np.sin(v)
        Z_s = np.cos(v)

        return go.Surface(
            x=X_s, y=Y_s, z=Z_s, opacity=0.1, colorscale="Greys", showscale=False
        )

    data = [
        # add_unit_sphere(),
        go.Scatter3d(
            x=Z[:, 0],
            y=Z[:, 1],
            z=Z[:, 2],
            opacity=0.8,
            mode="markers",
            marker=dict(size=3, color=y, colorscale="Viridis", opacity=0.8),
        )
    ]

    fig = go.Figure(data=data)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


def tsne_2D(
    model: Model,
    nb_speakers: int = 10,
    nb_samples: int = 150,
    init: Union[str, np.ndarray] = "random",
    speakers: List[str] = None,
) -> matplotlib.figure.Figure:
    Z, y = _filter_embeddings(model.embeddings, nb_speakers, nb_samples, speakers)

    Z_2d = TSNE(n_components=2, init=init).fit_transform(Z)

    df = pd.DataFrame()
    df["Speaker"] = y
    df["t-SNE_1"] = Z_2d[:, 0]
    df["t-SNE_2"] = Z_2d[:, 1]

    p = (
        ggplot(df, aes(x="t-SNE_1", y="t-SNE_2", color="Speaker"))
        + geom_point(alpha=0.5, show_legend=False)
        + geom_point(
            data=df.drop_duplicates(subset=["Speaker"]),
            mapping=aes(x="t-SNE_1", y="t-SNE_2", color="Speaker"),
            alpha=1,
            size=2,
        )
        + scale_color_hue()
        + labs(x=None, y=None)
        + theme_bw()
        + theme(
            figure_size=(8, 6),
            legend_title=element_text(size=10),
            legend_text=element_text(size=8),
            axis_title=element_blank(),
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            panel_grid=element_blank(),
            panel_border=element_blank(),
        )
    )

    return p, Z_2d


def pca_2D(
    model: Model,
    components: List[int] = [0, 1],
    speakers: Optional[List[str]] = None,
    nb_speakers: int = 6,
    nb_samples: int = 150,
) -> matplotlib.figure.Figure:
    Z, y = _filter_embeddings(model.embeddings, nb_speakers, nb_samples, speakers)

    n_components = max(components) + 1

    Z_2d = PCA(n_components=n_components).fit_transform(Z)

    df = pd.DataFrame(Z)
    df["Speaker"] = y
    df["PCA_1"] = Z_2d[:, components[0]]
    df["PCA_2"] = Z_2d[:, components[1]]

    p = (
        ggplot(df, aes(x="PCA_1", y="PCA_2", color="Speaker"))
        + geom_point(alpha=0.5, show_legend=False)
        + geom_point(
            data=df.drop_duplicates(subset=["Speaker"]),
            mapping=aes(x="PCA_1", y="PCA_2", color="Speaker"),
            alpha=1,
            size=2,
        )
        + scale_color_hue()
        + theme_bw()
        + theme(
            figure_size=(14, 8),
            legend_title=element_text(size=10),
            legend_text=element_text(size=8),
            axis_title=element_blank(),
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            panel_grid=element_blank(),
            panel_border=element_blank(),
        )
    )
    print(p)
