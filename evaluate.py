import argparse
import pickle
from pathlib import Path
import torch

from sslsv.utils.helpers import load_config, load_model
from sslsv.utils.evaluate import evaluate as evaluate_


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
    if not args.silent: print('Metrics', '=' * 35)
    for k, v in metrics.items():
        r = 2 if k.split('/')[-1] == 'eer' else 4
        print(k, round(v, r))

    if args.save:
        embeddings_save_path = checkpoint_dir + '/embeddings.pkl'
        with open(embeddings_save_path, 'wb') as f:
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
        '--save',
        action='store_true',
        help='Whether to save embeddings of test utterances.'
    )
    args = parser.parse_args()

    evaluate(args)
