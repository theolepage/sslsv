import torch
import pandas as pd

from sslsv.utils.helpers import load_config, load_model


def load_models(configs, override_names={}):
    models = {}

    for config_path in configs:
        config = load_config(config_path, verbose=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(config).to(device)
        
        checkpoint = torch.load(config.experiment_path / 'model_latest.pt')
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()

        model_name = config.experiment_name
        if config.experiment_name in override_names.keys():
            model_name = override_names[config.experiment_name]

        models[model_name] = {
            'model': model,
            'config': config,
            'device': device
        }
        
    return models


def evaluate_models(models, evaluation_cls, task_config, return_evals=False):
    evaluations = []

    for model_name, model_entry in models.items():
        evaluation = evaluation_cls(
            model=model_entry['model'],
            config=model_entry['config'],
            task_config=task_config,
            device=model_entry['device'],
            verbose=True,
            validation=False
        )

        models[model_name].update({
            'metrics': evaluation.evaluate(),
            'embeddings': evaluation.test_embeddings,
            'scores': evaluation.scores,
            'targets': evaluation.targets
        })

        evaluations.append(evaluation)

    if return_evals:
        return evaluations


def create_metrics_df(models):
    df = pd.DataFrame()

    for model_name, model_entry in models.items():
        metrics = {k:round(v, 4) for k, v in model_entry['metrics'].items()}
        df_ = pd.DataFrame({
            'Model': model_name,
            **metrics
        }, index=[0])
        df = pd.concat((df, df_))

    df = df.set_index('Model')
    df = df.sort_values(by=['voxceleb1_test_O/eer'], ascending=True)
    return df