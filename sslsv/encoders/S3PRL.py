from dataclasses import dataclass, field
from typing import List
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig

import s3prl
from s3prl.nn import S3PRLUpstream, Featurizer

from sslsv.encoders.TDNN import TDNN, TDNNConfig
from sslsv.encoders.ResNet34 import ResNet34, ResNet34Config
from sslsv.encoders.ECAPATDNN import ECAPATDNN, ECAPATDNNConfig


class MHFA(nn.Module):

    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256, nb_layers=13):
        super().__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(nb_layers), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layers), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k) # B, T, H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B, T, 1

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs


class WAP(nn.Module):

    def __init__(self, feat_dim, embed_dim, nb_layers):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(nb_layers))

        self.fc = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):
        # x: (N, D, T, F)

        x = torch.sum(x * F.softmax(self.weights, dim=0), dim=-1)
        # x: (N, D, T)

        x = x.mean(dim=-1) # x: (N, D)

        x = self.fc(x)
        
        return x


class GradMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class S3PRLPoolingEnum(Enum):
    """
    Enumeration representing pooling options for S3PRL encoder.

    Options:
        NONE (str): None.
        WAP (str): Weighted Average Pooling.
        MHFA (str): Multi-Head Factorized Attention Pooling.
        TDNN (str): TDNN encoder.
        RESNET (str): Fast ResNet-34 encoder.
        ECAPA (str): ECAPA-TDNN encoder.
    """

    NONE = "none"
    WAP = "wap"
    MHFA = "mhfa"
    TDNN = "tdnn"
    RESNET = "resnet"
    ECAPA = "ecapa"


@dataclass
class S3PRLConfig(BaseEncoderConfig):
    """
    S3PRL encoder configuration.

    Attributes:
        extract_mel_features (bool): Whether to extract Mel Spectrogram features.
        upstream (str): Name of the s3prl upstream.
        download_dir (str): Path to s3prl checkpoints.
        frozen (bool): Whether to freeze the encoder model.
        multilayer_features (bool): Whether to get features from all layers or the last one.
        avg_layer_features (bool): Whether to average features from all layers.
        global_feature_grad_mult (float): Factor applied to the gradients of the encoder.
        pooling (S3PRLPoolingEnum): Pooling method to aggregate s3prl features.
    """

    extract_mel_features: bool = False

    upstream: str = 'wavlm_base_plus'
    download_dir: str = 's3prl_hub'
    frozen: bool = True
    multilayer_features: bool = True
    avg_layer_features: bool = False
    global_feature_grad_mult: float = 1.0

    pooling: S3PRLPoolingEnum = S3PRLPoolingEnum.NONE


class S3PRL(BaseEncoder):
    """
    S3PRL encoder.

    Attributes:
    """

    def __init__(self, config: S3PRLConfig):
        """
        Initialize a S3PRL encoder.

        Args:
            config (S3PRLConfig): Encoder configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.frozen = config.frozen
        self.multilayer_features = config.multilayer_features
        self.avg_layer_features = config.avg_layer_features
        self.global_feature_grad_mult = config.global_feature_grad_mult

        if config.download_dir is not None:
            s3prl.util.download.set_dir(config.download_dir)

        self.upstream = S3PRLUpstream(config.upstream)
        if hasattr(self.upstream.upstream, "model"):
            if hasattr(self.upstream.upstream.model, "feature_grad_mult"):
                self.upstream.upstream.model.feature_grad_mult = 1.0
        self.upstream.eval()

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                if "mask_emb" in name:
                    param.requires_grad_(False)

        self.feat_dim = self.upstream.hidden_sizes[0]
        self.nb_layers = self.upstream.num_layers

        self.pooling = None
        if config.pooling == S3PRLPoolingEnum.NONE:
            self.pooling = nn.Identity()
        elif config.pooling == S3PRLPoolingEnum.WAP:
            self.pooling = WAP(
                feat_dim=self.feat_dim,
                embed_dim=self.encoder_dim,
                nb_layers=self.nb_layers
            )
        elif config.pooling == S3PRLPoolingEnum.MHFA:
            self.pooling = MHFA(
                head_nb=64,
                inputs_dim=self.feat_dim,
                outputs_dim=self.encoder_dim,
                nb_layers=self.nb_layers
            )
        elif config.pooling == S3PRLPoolingEnum.TDNN:
            self.pooling = TDNN(
                TDNNConfig(extract_mel_features=False, mel_n_mels=self.feat_dim)
            )
        elif config.pooling == S3PRLPoolingEnum.RESNET:
            self.pooling = ResNet34(
                ResNet34Config(extract_mel_features=False, mel_n_mels=self.feat_dim)
            )
        elif config.pooling == S3PRLPoolingEnum.ECAPA:
            self.pooling = ECAPATDNN(
                ECAPATDNNConfig(extract_mel_features=False, mel_n_mels=self.feat_dim)
            )
        else:
            raise Exception(f'Pooling method {config.pooling} is not handled')

        if self.avg_layer_features:
            self.layer_weights = nn.Parameter(torch.ones(self.nb_layers))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        X = super().forward(X)

        input_lengths = torch.full((X.size(0),), X.size(1), dtype=torch.long)

        if self.frozen:
            with torch.no_grad():
                Z, _ = self.upstream(X, input_lengths)
        else:
            Z, _ = self.upstream(X, input_lengths)

        Z = torch.stack(Z)

        Z = GradMultiply.apply(Z, self.global_feature_grad_mult)

        if not self.multilayer_features:
            Z = Z[-1]

        # Z: (F, N, T, D) or (N, T, D)

        if Z.ndim == 4:
            Z = Z.permute(1, 3, 2, 0) # (N, D, T, F)
            if self.avg_layer_features:
                Z = torch.sum(Z * F.softmax(self.layer_weights, dim=0), dim=-1)
        elif Z.ndim == 3:
            Z = Z.permute(0, 2, 1) # (N, D, T)

        Z = self.pooling(Z)
        
        return Z
