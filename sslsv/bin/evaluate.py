import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict

import argparse
import torch
import json

from sslsv.bin.train import MethodWrapper
from sslsv.utils.helpers import load_config, load_model, evaluate as evaluate_


def metrics_to_nested_dict(
    data: Dict[str, float],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Convert a flat dictionary of metrics data into a nested dictionary with
    task, dataset, and metric levels.

    Args:
        data (Dict[str, float]): Dictionary of metrics with the format 'test/task/dataset/metric'.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Nested dictionary with tasks as the first level,
            datasets as the second level, and metrics as the third level.

    Example:
        >>> metrics_to_nested_dict({
            'test/task1/dataset1/metric1': 10.0,
            'test/task1/dataset1/metric2': 20.0,
            'test/task2/dataset2/metric1': 15.0
        })
        {
            'task1': {
                'dataset1': {
                    'metric1': 10.0,
                    'metric2': 20.0
                }
            },
            'task2': {
                'dataset2': {
                    'metric1': 15.0
                }
            }
        }
    """
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
    """
    Print metrics in a nested format.

    Args:
        metrics (Dict[str, float]): Dictionary of metrics with the format 'test/task/dataset/metric'.

    Returns:
        None
    """
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
    """
    Evaluate a model from the CLI.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    config = load_config(args.config, verbose=not args.silent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)

    checkpoint = torch.load(config.model_ckpt_path / f"model_{args.model_suffix}.pt")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    if device == torch.device("cuda"):
        model = torch.nn.DataParallel(model)
    else:
        model = MethodWrapper(model)

    metrics = evaluate_(model, config, device, verbose=not args.silent)

    if args.silent:
        print(json.dumps(metrics))
    else:
        print_metrics(metrics)

    eval_path = config.model_path / "evaluation.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
    else:
        eval_data = {}

    eval_data.update(metrics)

    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_suffix",
        type=str,
        default="latest",
        help="Model checkpoint suffix (e.g. latest, avg, ...).",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to hide status messages and progress bars.",
    )
    args = parser.parse_args()

    evaluate(args)
