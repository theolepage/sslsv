import argparse
import pickle
from pathlib import Path
import torch

from sslsv.utils.helpers import load_config, load_model
from sslsv.utils.evaluate import evaluate as evaluate_


def evaluate(args):
    config, checkpoint_dir = load_config(args.config)
    model = load_model(config)

    checkpoint = torch.load(Path(checkpoint_dir) / 'model_latest.pt')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    model.eval()

    # Show metrics on speaker verification
    metrics, embeddings = evaluate_(model, config)
    print('Metrics:', metrics)

    # Save embeddings
    embeddings_save_path = checkpoint_dir + '/embeddings.pkl'
    with open(embeddings_save_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Speaker embeddings saved to {}'.format(embeddings_save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    evaluate(args)
