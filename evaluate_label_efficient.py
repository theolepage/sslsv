import argparse
from pathlib import Path

import torch
from torch import nn

from sslsv.losses.InfoNCE import InfoNCE
from sslsv.utils.helpers import load_config, load_dataloader, load_model
from sslsv.Trainer import Trainer


class Classifier(nn.Module):

    def __init__(self, model, add_last_layer=False):
        super().__init__()

        self.add_last_layer = add_last_layer
        self.model = model

        self.infonce = InfoNCE()
        if self.add_last_layer:
            self.classifier_fc = nn.Linear(1024, 1024)

    def forward(self, X, training=False):
        Z = self.model(X)
        if self.add_last_layer: Z = self.classifier_fc(Z)
        return Z

    def compute_loss(self, Z_1, Z_2):
        loss, accuracy = self.infonce((Z_1, Z_2))

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics


def get_checkpoint_name(checkpoint_dir, nb_labels_per_spk, fine_tune, supervised):
    checkpoint_dir += '_label-efficient-'
    checkpoint_dir += str(nb_labels_per_spk) + '-'
    if supervised:
        checkpoint_dir += 'supervised'
    else:
        checkpoint_dir += 'fine-tuned' if fine_tune else 'frozen'
    return checkpoint_dir


def train(args, nb_labels_per_spk, fine_tune=False, supervised=False):
    config, checkpoint_dir = load_config(args.config)

    config.data.wav_augment.enable = False
    config.training.optimizer = 'adam'
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.patience = args.patience
    config.training.learning_rate = args.lr
    if fine_tune: config.training.learning_rate /= 10

    train_dataloader = load_dataloader(config, nb_labels_per_spk)

    # Load model (to use as an encoder)
    model = load_model(config)
    if not supervised:
        checkpoint = torch.load(Path(checkpoint_dir) / 'model.pt')
        model.load_state_dict(checkpoint['model'])
        for p in model.parameters(): p.requires_grad = fine_tune

    # Create classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    add_last_layer = not fine_tune and not supervised
    classifier = Classifier(model, add_last_layer).to(device)
    classifier = torch.nn.DataParallel(classifier)

    checkpoint_dir = get_checkpoint_name(
        checkpoint_dir,
        nb_labels_per_spk,
        fine_tune,
        supervised
    )
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=classifier,
        train_dataloader=train_dataloader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    trainer.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    parser.add_argument(
        '--epochs',
        default=200,
        help='Number of epochs for trainings.'
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        help='Learning rate used during trainings.'
    )
    parser.add_argument(
        '--batch_size',
        default=64,
        help='Batch size used during trainings.'
    )
    parser.add_argument(
        '--patience',
        default=20,
        help='Number of epochs without a lower EER before ending training.'
    )
    parser.add_argument(
        '--nb_labels',
        default=[100, 50, 20, 10, 8, 6, 4, 2],
        nargs='*',
        help='Numbers of labels provided per speaker for each training.'
    )
    args = parser.parse_args()

    for nb_labels in args.nb_labels:
        train(args, int(nb_labels), supervised=True)
        train(args, int(nb_labels), fine_tune=False)
        train(args, int(nb_labels), fine_tune=True)
