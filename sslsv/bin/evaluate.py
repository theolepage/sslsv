import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict

import argparse
import torch
import json

from sslsv.bin.train import ModelWrapper
from sslsv.utils.helpers import load_config, load_model, evaluate as evaluate_


def metrics_to_nested_dict(
    data: Dict[str, float]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    res = {}

    tasks = set([k.split("/")[1] for k in data.keys() if k.split("/")[0] == "test"])

    for task in tasks:
        res[task] = {}

        datasets = set(
            [
                k.split("/")[2]
                for k in data.keys()
                if (k.split("/")[0] == "test" and k.split("/")[1] == task)
            ]
        )

        for dataset in datasets:
            res[task][dataset] = {}

            metrics = set(
                [
                    k.split("/")[3]
                    for k in data.keys()
                    if (
                        k.split("/")[0] == "test"
                        and k.split("/")[1] == task
                        and k.split("/")[2] == dataset
                    )
                ]
            )

            for metric in metrics:
                res[task][dataset][metric] = data[f"test/{task}/{dataset}/{metric}"]

    return res


def print_metrics(metrics: Dict[str, float]):
    metrics = metrics_to_nested_dict(metrics)

    for task in metrics.keys():
        print(f"\nEvaluation: {task}")
        for dataset in metrics[task].keys():
            print(f"  - {dataset}")
            for metric_name, metric_value in metrics[task][dataset].items():
                metric_value = round(metric_value, 2 if metric_name == "eer" else 4)
                space = " " * (
                    3
                    + max([len(k) for k in metrics[task][dataset].keys()])
                    - len(metric_name)
                )
                print(f"      {metric_name}:{space}{metric_value}")


def evaluate(args: argparse.Namespace):
    config = load_config(args.config, verbose=not args.silent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)

    checkpoint = torch.load(config.model_path / "model_latest.pt")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    if device == torch.device("cuda"):
        model = torch.nn.DataParallel(model)
    else:
        model = ModelWrapper(model)

    metrics = evaluate_(model, config, device, verbose=not args.silent)

    if args.silent:
        print(metrics)
    else:
        print_metrics(metrics)

    with open(config.model_path / "evaluation.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to model config file.")
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to hide status messages and progress bars.",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Whether to save embeddings of test utterances.",
    )
    args = parser.parse_args()

    evaluate(args)
