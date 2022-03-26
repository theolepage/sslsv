import argparse
import pickle
from pathlib import Path
import torch

from sslsv.utils.helpers import load_config, load_model
from sslsv.utils.evaluate import extract_embeddings, evaluate as evaluate_


def evaluate(args):
    config, checkpoint_dir = load_config(args.config)
    model = load_model(config)

    checkpoint = torch.load(Path(checkpoint_dir) / 'model.pt')
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    # Exract and save embeddings
    embeddings_save_path = checkpoint_dir + '/embeddings.pkl'
    embeddings = extract_embeddings(model, config.data)
    with open(embeddings_save_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Speaker embeddings saved to {}'.format(embeddings_save_path))

    # Show metrics on speaker verification
    test_metrics = evaluate_(embeddings, config.data.trials)
    print('Metrics:', test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    evaluate(args)
