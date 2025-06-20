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

    # train_files = list(Path(args.configs).rglob("training.json"))
    # for train_path in train_files:
    #     if "label-efficient" not in str(train_path):
    #         continue

    #     with open(train_path, "r") as file:
    #         train = json.load(file)

    #     model_name = str(train_path)[7:-14]

    #     if "99" not in train:
    #         continue

    #     res[model_name] = {
    #         "VoxCeleb1-O___EER (%)": formatf2(
    #             train["99"]["val/sv_cosine/voxceleb1_test_O/eer"]
    #         ),
    #         "VoxCeleb1-O___minDCF": formatf4(
    #             train["99"]["val/sv_cosine/voxceleb1_test_O/mindcf"]
    #         ),
    #     }

    # print(json.dumps(res, indent=4))
    # return

    configs = list(Path(args.configs).rglob("config.yml"))

    for config_path in configs:
        eval_path = config_path.with_name("evaluation.json")
        train_path = config_path.with_name("training.json")

        if not eval_path.is_file():
            # print(f"No evaluation file found for {config_path}")
            continue

        with open(eval_path, "r") as file:
            eval = json.load(file)

        # with open(train_path, "r") as file:
        #     train = json.load(file)

        model_name = str(config_path)[7:-11]

        res[model_name] = {}

        if "test/sv_cosine/voxceleb1_test_O/eer" in eval:
            res[model_name] = {
                "VoxCeleb1-O___EER (%)": formatf2(
                    eval["test/sv_cosine/voxceleb1_test_O/eer"]
                ),
                "VoxCeleb1-O___minDCF": formatf4(
                    eval["test/sv_cosine/voxceleb1_test_O/mindcf"]
                ),
            }

        if "test/sv_cosine/voxceleb1_test_E/eer" in eval:
            res[model_name].update(
                {
                    "VoxCeleb1-E___EER (%)": formatf2(
                        eval["test/sv_cosine/voxceleb1_test_E/eer"]
                    ),
                    "VoxCeleb1-E___minDCF": formatf4(
                        eval["test/sv_cosine/voxceleb1_test_E/mindcf"]
                    ),
                }
            )

        if "test/sv_cosine/voxceleb1_test_H/eer" in eval:
            res[model_name].update(
                {
                    "VoxCeleb1-H___EER (%)": formatf2(
                        eval["test/sv_cosine/voxceleb1_test_H/eer"]
                    ),
                    "VoxCeleb1-H___minDCF": formatf4(
                        eval["test/sv_cosine/voxceleb1_test_H/mindcf"]
                    ),
                }
            )

        if "test/sv_cosine/sitw_eval_core-core/eer" in eval:
            res[model_name].update(
                {
                    "SITW___EER (%)": formatf2(
                        eval["test/sv_cosine/sitw_eval_core-core/eer"]
                    ),
                    "SITW___minDCF": formatf4(
                        eval["test/sv_cosine/sitw_eval_core-core/mindcf"]
                    ),
                }
            )

        if "test/sv_cosine/voices2019_dev/eer" in eval:
            res[model_name].update(
                {
                    "VOiCES___EER (%)": formatf2(
                        eval["test/sv_cosine/voices2019_dev/eer"]
                    ),
                    "VOiCES___minDCF": formatf4(
                        eval["test/sv_cosine/voices2019_dev/mindcf"]
                    ),
                }
            )

        if "test/classification/voxlingua107_train/accuracy" in eval:
            res[model_name].update(
                {
                    "VoxLingua107___Accuracy": round(
                        eval["test/classification/voxlingua107_train/accuracy"] * 100, 2
                    ),
                    "VoxLingua107___F1 score": round(
                        eval["test/classification/voxlingua107_train/f1_score"] * 100, 2
                    ),
                }
            )

        if "test/classification/cremad_train/accuracy" in eval:
            res[model_name].update(
                {
                    "CREMA-D___Accuracy": round(
                        eval["test/classification/cremad_train/accuracy"] * 100, 2
                    ),
                    "CREMA-D___F1 score": round(
                        eval["test/classification/cremad_train/f1_score"] * 100, 2
                    ),
                }
            )

        if "test/classification/voxceleb1_train_speaker/accuracy" in eval:
            res[model_name].update(
                {
                    "VoxCeleb1-Speaker___Accuracy": round(
                        eval["test/classification/voxceleb1_train_speaker/accuracy"]
                        * 100,
                        2,
                    ),
                    "VoxCeleb1-Speaker___F1 score": round(
                        eval["test/classification/voxceleb1_train_speaker/f1_score"]
                        * 100,
                        2,
                    ),
                }
            )

        if "test/classification/voxceleb1_train_gender/accuracy" in eval:
            res[model_name].update(
                {
                    "VoxCeleb1-Gender___Accuracy": round(
                        eval["test/classification/voxceleb1_train_gender/accuracy"]
                        * 100,
                        2,
                    ),
                    "VoxCeleb1-Gender___F1 score": round(
                        eval["test/classification/voxceleb1_train_gender/f1_score"]
                        * 100,
                        2,
                    ),
                }
            )

        # last_epoch = list(train.keys())[-1]

        # if "ssps_speaker_acc" in train[last_epoch]:
        #     res[model_name].update(
        #         {
        #             "Speaker Accuracy": formatf2(train[last_epoch]["ssps_speaker_acc"]),
        #             "Video Accuracy": formatf2(train[last_epoch]["ssps_video_acc"]),
        #         }
        #     )

        # if "ssps_kmeans_nmi_speaker" in train[last_epoch]:
        #     res[model_name].update(
        #         {
        #             "NMI Speaker": formatf2(train[last_epoch]["ssps_kmeans_nmi_speaker"]),
        #             "NMI Video": formatf2(train[last_epoch]["ssps_kmeans_nmi_video"]),
        #         }
        #     )

    # print(json.dumps(res, indent=4))
    print(json.dumps(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, help="Path to models folder.")
    args = parser.parse_args()

    export_model_metrics(args)
