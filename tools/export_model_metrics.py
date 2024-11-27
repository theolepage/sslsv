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

        eer = eval["test/sv_cosine/voxceleb1_test_O/eer"]
        mindcf = eval["test/sv_cosine/voxceleb1_test_O/mindcf"]

        res[model_name] = {
            "EER (%)": f"{eer:.2f}",
            "minDCF": f"{mindcf:.4f}",
        }

        last_epoch = list(train.keys())[-1]

        if "ssps_speaker_acc" in train[last_epoch]:
            ssps_speaker_acc = train[last_epoch]["ssps_speaker_acc"]
            ssps_video_acc = train[last_epoch]["ssps_video_acc"]
            ssps_kmeans_nmi_speaker = train[last_epoch]["ssps_kmeans_nmi_speaker"]
            ssps_kmeans_nmi_video = train[last_epoch]["ssps_kmeans_nmi_video"]

            res[model_name].update(
                {
                    "EER (%)": f"{eer:.2f}",
                    "minDCF": f"{mindcf:.4f}",
                    "Speaker Accuracy": f"{ssps_speaker_acc:.2f}",
                    "Video Accuracy": f"{ssps_video_acc:.2f}",
                    "NMI Speaker": f"{ssps_kmeans_nmi_speaker:.2f}",
                    "NMI Video": f"{ssps_kmeans_nmi_video:.2f}",
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
