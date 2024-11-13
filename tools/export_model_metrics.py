import json
from pathlib import Path
import argparse


def export_model_metrics(args: argparse.Namespace):
    """
    Export model metrics.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    res = {}

    for config_path in args.configs:
        eval_path = Path(config_path).with_name("evaluation.json")

        if not eval_path.is_file():
            print(f"No evaluation file found for {config_path}")
            continue

        with open(eval_path, "r") as file:
            eval = json.load(file)

        # model_name = config_path.split("/")[-2]
        model_name = "/".join(config_path.split("/")[-3:-1])

        eer = float(eval["test/sv_cosine/voxceleb1_test_O/eer"])
        mindcf = float(eval["test/sv_cosine/voxceleb1_test_O/mindcf"])

        res[model_name] = {
            "EER (%)": f"{eer:.2f}",
            "minDCF": f"{mindcf:.4f}",
        }

    print()
    # print(json.dumps(res, indent=4))
    print(json.dumps(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Path to models config file.")
    args = parser.parse_args()

    export_model_metrics(args)
