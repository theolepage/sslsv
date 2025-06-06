import torch

import numpy as np

import json

from sslsv.evaluations.CosineSVEvaluation import (
    CosineSVEvaluation,
    CosineSVEvaluationTaskConfig,
)
from sslsv.utils.helpers import load_config, load_model


class _CosineSVEvaluation(CosineSVEvaluation):

    def __init__(self, embeddings_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if embeddings_path:
            self.embeddings_path = str(self.config.model_path / embeddings_path)
        else:
            self.embeddings_path = None

    def _extract_test_embeddings(self, trials):
        if self.embeddings_path:
            test_embeddings = torch.load(self.embeddings_path)
            self.test_embeddings = {
                k.replace("data/", ""): v for k, v in test_embeddings.items()
            }
        else:
            super()._extract_test_embeddings(trials)


def evaluate_sv(
    models,
    embeddings_path,
    trials=["voxceleb1_test_O"],
    checkpoint_suffix="avg",
):
    res = {}

    for name, path in models.items():
        config = load_config(f"{path}/config.yml", verbose=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(config).to(device)

        checkpoint = torch.load(
            config.model_ckpt_path / f"model_{checkpoint_suffix}.pt", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()

        evaluation = _CosineSVEvaluation(
            model=model,
            config=config,
            task_config=CosineSVEvaluationTaskConfig(
                trials=trials,
                metrics=["eer", "mindcf", "cllr"],
                batch_size=16,
            ),
            device=device,
            verbose=True,
            validation=False,
            embeddings_path=embeddings_path,
        )
        metrics = evaluation.evaluate()

        res[name] = {
            "config": config,
            "scores": evaluation.scores,
            "targets": evaluation.targets,
            "evaluation": evaluation,
        }

        metrics = {k: round(v, 4) for k, v in metrics.items()}
        print(name, json.dumps(metrics, indent=4))

    return res


import torch
import pandas as pd

from plotnine import (
    ggplot,
    aes,
    labs,
    theme_bw,
    theme,
    element_text,
    scale_y_continuous,
    geom_line,
    geom_boxplot,
    geom_point,
)

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict


def plot_intra_class_similarity(type, checkpoints):
    data = pd.DataFrame({})

    if type == "speaker":
        key = -3
    elif type == "video":
        key = -2
    else:
        raise Exception()

    for method_name, ckpt_path in checkpoints.items():
        embeddings = torch.load(ckpt_path)

        # Determine set of embeddings for each speaker
        classes = defaultdict(list)
        for path, embedding in embeddings.items():
            classes[path.split("/")[key]].append(embedding)

        similarities = []
        for class_embeddings in classes.values():
            class_embeddings = torch.cat(class_embeddings, dim=0)
            class_embeddings = torch.nn.functional.normalize(
                class_embeddings, p=2, dim=1
            )
            class_sim = class_embeddings @ class_embeddings.T
            mask = torch.eye(class_sim.size(0), dtype=torch.bool)
            class_sim = class_sim[~mask].mean().item()
            similarities.append(class_sim)

        data[method_name] = similarities

    data = data.melt(var_name="method", value_name="similarity")
    data["method"] = pd.Categorical(
        data["method"], categories=checkpoints.keys(), ordered=True
    )

    p = (
        ggplot(data, aes(x="method", y="similarity", fill="method"))
        + geom_boxplot(outlier_alpha=0.5)
        # + labs(x=None, y=None, title=f"Intra-{type} cosine similarity")
        + labs(x="", y="")
        + scale_y_continuous(
            limits=(0.3, 0.7),
            breaks=np.round(np.arange(0.3, 0.8, 0.1), 1),
            labels=[" {:.2f}".format(x) for x in np.round(np.arange(0.3, 0.8, 0.1), 1)],
        )
        # + scale_y_continuous(limits=(0.25, 0.75), breaks=[0.25, 0.5, 0.75])
        + theme_bw()
        + theme(
            figure_size=(8, 4.75), text=element_text(size=14), legend_position="none"
        )
    )

    stats = (
        data.groupby("method")["similarity"]
        .agg(["mean", "min", "max", "median"])
        .reset_index()
    )

    return p, stats


def plot_intra_class_similarity_by_class(type, checkpoints, nb_classes=10):
    data = pd.DataFrame({})

    if type == "speaker":
        key = -3
    elif type == "video":
        key = -2
    else:
        raise Exception()

    torch.manual_seed(0)
    idx = None

    for method_name, ckpt_path in checkpoints.items():
        embeddings = torch.load(ckpt_path)

        # Determine set of embeddings for each class
        classes = defaultdict(list)
        for path, embedding in embeddings.items():
            classes[path.split("/")[key]].append(embedding)

        if idx is None:
            idx = torch.randint(0, len(classes), size=(nb_classes,))
        classes = {k: v for i, (k, v) in enumerate(classes.items()) if i in idx}

        similarities = []
        for class_embeddings in classes.values():
            class_embeddings = torch.cat(class_embeddings, dim=0)
            class_embeddings = torch.nn.functional.normalize(
                class_embeddings, p=2, dim=1
            )
            class_sim = class_embeddings @ class_embeddings.T
            mask = torch.eye(class_sim.size(0), dtype=torch.bool)
            class_sim = class_sim[~mask].mean().item()
            similarities.append(class_sim)

        temp_df = pd.DataFrame(
            {
                "class": list(classes.keys()),
                "similarity": similarities,
                "method": method_name,
            }
        )
        data = pd.concat([data, temp_df], ignore_index=True)

    data["method"] = pd.Categorical(
        data["method"], categories=checkpoints.keys(), ordered=True
    )

    p = (
        ggplot(data, aes(x="class", y="similarity", color="method", group="method"))
        + geom_line()
        + labs(
            x="Class",
            y="Similarity",
            title=f"Intra-{type} cosine similarity by {type}",
        )
        + theme_bw()
        + theme(
            figure_size=(12, 8),
            text=element_text(size=14),
            axis_text_x=element_text(angle=45, ha="right"),
        )
    )

    return p


def plot_inter_class_similarity(type, checkpoints, nb_samples=1000):
    data = pd.DataFrame({})

    if type == "speaker":
        key = -3
    elif type == "video":
        key = -2
    else:
        raise Exception()

    torch.manual_seed(0)
    idx = None

    for method_name, ckpt_path in checkpoints.items():
        checkpoint = torch.load(ckpt_path)

        checkpoint = {k: checkpoint[k] for k in sorted(checkpoint)}

        embeddings = torch.cat(list(checkpoint.values()), dim=0)

        labels = [v.split("/")[key] for v in checkpoint.keys()]
        labels = torch.tensor(LabelEncoder().fit_transform(labels))

        # Sample
        if idx is None:
            idx = torch.randint(0, len(embeddings), size=(nb_samples,))
        embeddings = embeddings[idx]
        labels = labels[idx]

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarities = embeddings @ embeddings.T

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        similarities = similarities[~mask]

        data[method_name] = similarities.tolist()

    data = data.melt(var_name="method", value_name="similarity")
    data["method"] = pd.Categorical(
        data["method"], categories=checkpoints.keys(), ordered=True
    )

    p = (
        ggplot(data, aes(x="method", y="similarity", fill="method"))
        + geom_boxplot(outlier_alpha=0.5)
        # + labs(x=None, y=None, title=f"Inter-{type} cosine similarity")
        + labs(x="", y="")
        + scale_y_continuous(limits=(-0.5, 0.75))
        # + scale_y_continuous(limits=(0.3, 0.7), breaks=[0.3, 0.4, 0.5, 0.6, 0.7])
        + theme_bw()
        + theme(
            figure_size=(8, 4.75), text=element_text(size=14), legend_position="none"
        )
    )

    stats = (
        data.groupby("method")["similarity"]
        .agg(["mean", "min", "max", "median"])
        .reset_index()
    )

    return p, stats


def plot_inter_speaker_center_similarity(checkpoints):
    data = pd.DataFrame({})

    for method_name, ckpt_path in checkpoints.items():
        embeddings = torch.load(ckpt_path)

        # Determine set of embeddings for each speaker
        classes = defaultdict(list)
        for path, embedding in embeddings.items():
            classes[path.split("/")[-3]].append(embedding)

        # Determien average embedding for each speaker
        classes_center = []
        for v in classes.values():
            classes_center.append(torch.cat(v, dim=0).mean(dim=0, keepdim=True))
        classes_center = torch.cat(classes_center, dim=0)

        classes_center = torch.nn.functional.normalize(classes_center, p=2, dim=1)
        similarities = classes_center @ classes_center.T

        mask = torch.eye(similarities.size(0), dtype=torch.bool)
        similarities = similarities[~mask]

        data[method_name] = similarities.tolist()

    data = data.melt(var_name="method", value_name="similarity")
    data["method"] = pd.Categorical(
        data["method"], categories=checkpoints.keys(), ordered=True
    )

    p = (
        ggplot(data, aes(x="method", y="similarity"))
        + geom_boxplot(outlier_alpha=0)
        + labs(x=None, y=None, title=f"Inter-speaker cosine similarity")
        + theme_bw()
        + theme(figure_size=(12, 8), text=element_text(size=14))
    )

    stats = (
        data.groupby("method")["similarity"]
        .agg(["mean", "min", "max", "median"])
        .reset_index()
    )

    return p, stats
