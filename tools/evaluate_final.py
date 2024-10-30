import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import json
import subprocess
from tqdm import tqdm


def evaluate(args: argparse.Namespace):
    """
    Evaluate models.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    res = {}

    for config_path in tqdm(args.configs):
        # Model average
        subprocess.run(
            [
                "python",
                "sslsv/bin/average_model.py",
                config_path,
                # "--count",
                # "5",
                # "--limit_nb_epochs",
                # "10",
                "--silent",
            ]
        )

        # Create eval config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config["evaluation"]["test"][0].update(
            {
                "num_frames": 1,
                "batch_size": 1,
                "frame_length": None,
            }
        )
        config_eval_path = config_path.replace("config.yml", "config_eval.yml")
        with open(config_eval_path, "w") as f:
            yaml.dump(config, f)

        # Evaluate
        eval = subprocess.run(
            [
                "./evaluate_ddp.sh",
                "2",
                config_eval_path,
                "--model_suffix",
                "avg",
                "--silent",
            ],
            capture_output=True,
            text=True,
        )

        # with open(config_path.replace("config.yml", "training.json")) as f:
        #     training = json.load(f)

        eval = json.loads(eval.stdout.strip())

        res[config_path.split("/")[-2]] = {
            "EER (%)": str(
                round(float(eval["test/sv_cosine/voxceleb1_test_O/eer"]), 2)
            ),
            "minDCF": str(
                round(float(eval["test/sv_cosine/voxceleb1_test_O/mindcf"]), 4)
            ),
            # "Speaker Accuracy": str(
            # round(float(training["109"]["ssps_speaker_acc"]), 4)
            # ),
            # "Video Accuracy": str(round(float(training["109"]["ssps_video_acc"]), 4)),
            # "Coverage": str(round(float(training["109"]["ssps_coverage"]), 4)),
            # "NMI": str(round(float(training["109"]["ssps_kmeans_nmi"]), 4)),
            # "ARI": str(round(float(training["109"]["ssps_kmeans_ari"]), 4)),
        }

        # Delete eval config
        os.remove(config_eval_path)

    # Export metrics for google sheet
    print()
    print(json.dumps(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Path to models config file.")
    args = parser.parse_args()

    evaluate(args)
