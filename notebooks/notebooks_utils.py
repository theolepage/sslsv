from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import pandas as pd

from sslsv.Config import Config
from sslsv.methods._BaseMethod import BaseMethod
from sslsv.evaluations._BaseEvaluation import BaseEvaluation, EvaluationTaskConfig
from sslsv.utils.helpers import load_config, load_model


@dataclass
class Model:

    model: BaseMethod
    config: Config
    device: torch.device


def load_models(
    configs: List[str],
    override_names: Dict[str, str] = {},
) -> Dict[str, Model]:
    models = {}

    for config_path in configs:
        config = load_config(config_path, verbose=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(config).to(device)

        checkpoint = torch.load(config.model_path / "model_latest.pt")
        model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()

        model_name = config.model_name
        if model_name in override_names.keys():
            model_name = override_names[model_name]

        models[model_name] = Model(model, config, device)

    return models


def evaluate_models(
    models: Dict[str, Model],
    evaluation_cls: BaseEvaluation,
    task_config: EvaluationTaskConfig,
    return_evals: bool = False,
) -> Optional[List[BaseEvaluation]]:
    evaluations = []

    for model_name, model_entry in models.items():
        evaluation = evaluation_cls(
            model=model_entry["model"],
            config=model_entry["config"],
            task_config=task_config,
            device=model_entry["device"],
            verbose=True,
            validation=False,
        )

        models[model_name].update(
            {
                "metrics": evaluation.evaluate(),
                "embeddings": evaluation.test_embeddings,
                "scores": evaluation.scores,
                "targets": evaluation.targets,
            }
        )

        evaluations.append(evaluation)

    if return_evals:
        return evaluations


def create_metrics_df(models: Dict[str, Model]) -> pd.DataFrame:
    df = pd.DataFrame()

    for model_name, model_entry in models.items():
        metrics = {k: round(v, 4) for k, v in model_entry["metrics"].items()}
        df_ = pd.DataFrame({"Model": model_name, **metrics}, index=[0])
        df = pd.concat((df, df_))

    df = df.set_index("Model")
    df = df.sort_values(by=["test/sv_cosine/voxceleb1_test_O/eer"], ascending=True)
    return df
