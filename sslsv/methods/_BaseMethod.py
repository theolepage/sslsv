import torch.nn as nn

from dataclasses import dataclass


@dataclass
class BaseMethodConfig:

    pass


class BaseMethod(nn.Module):

    def __init__(self, config, create_encoder_fn):
        super().__init__()

        self.config = config

        self.encoder = create_encoder_fn()

    def forward(self, X, training=False):
        return self.encoder(X)

    def get_learnable_params(self):
        return [{'params': self.encoder.parameters()}]

    def update_optim(
        self,
        optimizer,
        training_config,
        step,
        nb_steps,
        nb_steps_per_epoch
    ):
        # Equivalent to StepLR(..., step_size=5, gamma=0.95)
        init_lr = training_config.learning_rate
        lr = init_lr * (0.95 ** ((step // nb_steps_per_epoch) // 5))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_step(self, Z, labels, step, samples):
        raise NotImplementedError

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_train_epoch_start(self, epoch, max_epochs):
        pass

    def on_train_epoch_end(self, epoch, max_epochs):
        pass

    def on_train_step_start(self, step, max_steps):
        pass

    def on_train_step_end(self, step, max_steps):
        pass

    def on_before_backward(self):
        pass

    def on_after_backward(self):
        pass