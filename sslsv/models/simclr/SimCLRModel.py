import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders.ThinResNet34 import ThinResNet34

from sslsv.losses.InfoNCE import InfoNCE
from sslsv.losses.VICReg import VICReg
from sslsv.losses.BarlowTwins import BarlowTwins


class SimCLRModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.enable_mlp = config.enable_mlp
        self.mlp_dim = config.mlp_dim
        self.infonce_weight = config.infonce_weight
        self.vicreg_weight = config.vicreg_weight
        self.barlowtwins_weight = config.barlowtwins_weight
        self.representations_losses = config.representations_losses
        self.embeddings_losses = config.embeddings_losses

        self.infonce = InfoNCE()
        self.vicreg = VICReg(
            config.vic_inv_weight,
            config.vic_var_weight,
            config.vic_cov_weight
        )
        self.barlowtwins = BarlowTwins(
            config.barlowtwins_lambda
        )

        self.encoder = ThinResNet34()
        self.mlp = nn.Sequential(
            nn.Linear(1024, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim)
        )

    def forward(self, X, training=False):
        Y = self.encoder(X)

        # Do not use projector for inference / evaluaton
        if not training: return Y 

        Z = self.mlp(Y) if self.enable_mlp else None
        
        return Y, Z

    def compute_loss_(self, Z_1, Z_2, losses):
        loss, accuracy = 0, 0
        if losses[0]: # infonce
            loss, accuracy = self.infonce((Z_1, Z_2))
            loss = self.infonce_weight * loss
        if losses[1]: # vicreg
            loss += self.vicreg_weight * self.vicreg((Z_1, Z_2))
        if losses[2]: # barlowtwins
            loss += self.barlowtwins_weight * self.barlowtwins((Z_1, Z_2))
        return loss, accuracy

    def compute_loss(self, Z_1, Z_2):
        Y_1, Z_1 = Z_1
        Y_2, Z_2 = Z_2

        metrics = {}

        # Representations losses
        Y_loss, Y_accuracy = self.compute_loss_(
            Y_1,
            Y_2,
            self.representations_losses
        )
        loss = Y_loss
        metrics['train_Y_loss'] = Y_loss
        metrics['train_Y_accuracy'] = Y_accuracy

        # Embeddings losses
        if self.enable_mlp:
            Z_loss, Z_accuracy = self.compute_loss_(
                Z_1,
                Z_2,
                self.embeddings_losses
            )
            loss += Z_loss
            metrics['train_Z_loss'] = Z_loss
            metrics['train_Z_accuracy'] = Z_accuracy

        metrics['train_loss'] = loss

        return loss, metrics
