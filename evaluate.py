import argparse
import pickle
from pathlib import Path
import torch
import json

from sslsv.utils.helpers import load_config, load_model
from sslsv.utils.evaluate import evaluate as evaluate_


def print_metrics(metrics):
    test_sets = set([k.split('/')[1] for k in metrics.keys()])
    for test_set in test_sets:
        print(f'- {test_set}')
        for k, v in metrics.items():
            metric_name = k.split('/')[2]
            precision = 2 if metric_name == 'eer' else 4
            space = ' ' * (
                3 +
                max([len(k.split('/')[2]) for k in metrics.keys()]) -
                len(metric_name)
            )
            print(f'    {metric_name}:{space}{round(v, precision)}')


def evaluate(args):
    config, checkpoint_dir = load_config(
        args.config,
        verbose=not args.silent
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(config).to(device)

    checkpoint = torch.load(Path(checkpoint_dir) / 'model_latest.pt')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    model = torch.nn.DataParallel(model)

    metrics, embeddings = evaluate_(
        model,
        config,
        device,
        verbose=not args.silent
    )

    if args.silent:
        print(metrics)
    else:
        print_metrics(metrics)

    with open(Path(checkpoint_dir) / 'evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    if args.save_embeddings:
        with open(Path(checkpoint_dir) / 'embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        help='Path to model config file.'
    )
    parser.add_argument(
        '--silent',
        action='store_true',
        help='Whether to hide status messages and progress bars.'
    )
    parser.add_argument(
        '--save_embeddings',
        action='store_true',
        help='Whether to save embeddings of test utterances.'
    )
    args = parser.parse_args()

    evaluate(args)
