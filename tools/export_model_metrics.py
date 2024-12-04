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
    def formatf2(value):
        return f"{value:.2f}"


    def formatf4(value):
        return f"{value:.4f}"

    res = {}

    for config_path in args.configs:
        eval_path = Path(config_path).with_name("evaluation.json")
        train_path = Path(config_path).with_name("training.json")

        if not eval_path.is_file():
            print(f"No evaluation file found for {config_path}")
            continue

        with open(eval_path, "r") as file:
            eval = json.load(file)

        with open(train_path, "r") as file:
            train = json.load(file)

        # model_name = config_path.split("/")[-2]
        model_name = "/".join(config_path.split("/")[-3:-1])

        res[model_name] = {
            "VoxCeleb1-O___EER (%)": formatf2(eval["test/sv_cosine/voxceleb1_test_O/eer"]),
            "VoxCeleb1-O___minDCF": formatf4(eval["test/sv_cosine/voxceleb1_test_O/mindcf"]),
            "VoxCeleb1-E___EER (%)": formatf2(eval["test/sv_cosine/voxceleb1_test_E/eer"]),
            "VoxCeleb1-E___minDCF": formatf4(eval["test/sv_cosine/voxceleb1_test_E/mindcf"]),
            "VoxCeleb1-H___EER (%)": formatf2(eval["test/sv_cosine/voxceleb1_test_H/eer"]),
            "VoxCeleb1-H___minDCF": formatf4(eval["test/sv_cosine/voxceleb1_test_H/mindcf"]),
        }

        last_epoch = list(train.keys())[-1]

        if "ssps_speaker_acc" in train[last_epoch]:
            res[model_name].update(
                {
                    "Speaker Accuracy": formatf2(train[last_epoch]["ssps_speaker_acc"]),
                    "Video Accuracy": formatf2(train[last_epoch]["ssps_video_acc"])",
                    "NMI Speaker": formatf2(train[last_epoch]["ssps_kmeans_nmi_speaker"]),
                    "NMI Video": formatf2(train[last_epoch]["ssps_kmeans_nmi_video"]),
                }
            )

    print()
    # print(json.dumps(res, indent=4))
    print(json.dumps(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Path to models config file.")
    args = parser.parse_args()

    export_model_metrics(args)
