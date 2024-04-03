from dataclasses import dataclass
from typing import Union

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler as TorchSampler


@dataclass
class SamplerConfig:

    enable: bool = True
    nb_samples_per_spk: Union[int, None] = None
    create_contrastive_pairs: bool = False
    prevent_class_collisions: bool = False
    randomize_at_each_epoch: bool = False


class Sampler(TorchSampler):

    def __init__(self, dataset, batch_size, config, seed=0):
        self.labels = dataset.labels
        self.batch_size = batch_size
        self.config = config
        self.seed = seed

        self.epoch = 0
        self.count = 0

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.config.randomize_at_each_epoch:
            self.seed = self.seed + self.epoch
        rng = np.random.default_rng(seed=self.seed)

        indices = rng.permutation(len(self.labels))

        # Create list of utterances for each speaker
        spk_to_utterances = defaultdict(list)
        for i in indices:
            spk_to_utterances[self.labels[i]].append(int(i))
        speakers = list(spk_to_utterances.keys())
        speakers.sort()

        # Create indexes by providing pairs if create_contrastive_pairs
        # is set to True
        x = []
        y = []
        for i, speaker in enumerate(speakers):
            utterances = spk_to_utterances[speaker]

            nb_utt = len(utterances)

            if self.config.nb_samples_per_spk:
                nb_utt = min(nb_utt, self.config.nb_samples_per_spk)

            if self.config.create_contrastive_pairs:
                nb_utt = nb_utt - nb_utt % 2
                for idx in range(0, nb_utt, 2):
                    x.append((utterances[idx], utterances[idx + 1]))
                    y.append(i)
                    x.append((utterances[idx + 1], utterances[idx]))
                    y.append(i)
            else:
                for idx in range(nb_utt):
                    x.append(utterances[idx])
                    y.append(i)

        # Shuffle indices and avoid having two samples of the same speaker
        # in the same batch if prevent_class_collisions is set to True
        x_ = []
        y_ = []
        for i in rng.permutation(len(y)):
            batch_start_i = len(y_) - len(y_) % self.batch_size
            if (
                self.config.prevent_class_collisions and
                y[i] in y_[batch_start_i:]
            ):
                continue

            x_.append(x[i])
            y_.append(y[i])

        self.count = len(x_) - len(x_) % self.batch_size
        return iter(x_[:self.count])
