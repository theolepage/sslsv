from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import numpy as np

from torchaudio.transforms import MelSpectrogram

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMomentumMethod import (
    BaseMomentumMethod,
    BaseMomentumMethodConfig,
    initialize_momentum_params,
)
from sslsv.utils.distributed import get_world_size

from .DINOSKLoss import DINOSKLoss


class DINOSKHead(nn.Module):
    """
    Head module for DINOSK.

    Attributes:
        mlp (nn.Sequential): MLP module.
        prototypes (Optional[nn.utils.weight_norm]): Last layer module.
    """

    ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nb_prototypes: int,
        activation: str = "relu",
        sdpn_init: bool = False,
        include_prototypes: bool = True,
    ):
        """
        Initialize a DINOSK head module.

        Args:
            input_dim (int): Dimension of the input.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output.
            nb_prototypes (int): Number of prototypes.
            activation (str): Activation function of the MLP ('relu' or 'gelu').
            sdpn_init (bool): Whether to initialize the MLP as SDPN does
                (truncated normal with std=0.02 and zeroed biases).
            include_prototypes (bool): Whether the head owns its prototypes.
                Disabled when prototypes are shared with the momentum branch.

        Returns:
            None
        """
        super().__init__()

        act_fn = self.ACTIVATIONS[activation]

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, output_dim),
        )

        if sdpn_init:
            self.head.apply(self._init_weights)

        self.prototypes = None
        if include_prototypes:
            self.prototypes = nn.utils.weight_norm(
                nn.Linear(output_dim, nb_prototypes, bias=False)
            )
            self.prototypes.weight_g.data.fill_(1)
            self.prototypes.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module):
        """
        Initialize the weights of a Linear module (SDPN initialization).

        Args:
            m (nn.Module): Module to initialize.

        Returns:
            None
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: T) -> T:
        """
        Forward pass.

        Args:
            x (T): Input tensor.

        Returns:
            T: Output tensor. Prototype scores, or the L2-normalized embeddings
                when the prototypes are held by the parent method.
        """
        x = self.head(x)
        x = F.normalize(x, p=2, dim=-1)
        if self.prototypes is not None:
            x = self.prototypes(x)
        return x


@dataclass
class DINOSKConfig(BaseMomentumMethodConfig):
    """
    DINOSK method configuration.

    Attributes:
        global_count (int): Number of global (large) views.
        local_count (int): Number of local (small) views.
        start_tau (float): Initial value for tau (momentum parameters update).
        head_hidden_dim (int): Head hidden dimension.
        head_output_dim (int): Head output dimension.
        nb_prototypes (int): Number of prototypes.
        student_temperature (float): Temperature value for the student.
        teacher_temperature (float): Temperature value for the teacher.
        memax_weight (float): Weight for Me-Max regularization.
        koleo_weight (float): Weight for Koleo regularization.
        shared_prototypes (bool): Whether the student and teacher branches score
            against a single prototype bank trained by SGD (SDPN) instead of the
            student holding its own bank and the teacher an EMA copy.
        prototypes_grad_world_size_sum (bool): Whether to sum the prototypes
            gradients across processes instead of averaging them (SDPN).
        head_activation (str): Activation function of the head MLP ('relu' or 'gelu').
        sdpn_init (bool): Whether to initialize the head MLP as SDPN does.
        prototypes_weight_decay (bool): Whether to apply weight decay to the prototypes.
        sdpn_views (bool): Whether to assemble views as SDPN does: local frames arrive
            concatenated by pairs, every row is trimmed by `sdpn_views_trim` samples to
            yield an even number of Mel frames, and each pair is split in half only
            after the Mel transform. Requires `ssl_dino_local_pairs` on the dataset and
            `extract_mel_features: false` on the encoder, as the Mel transform is then
            held by the method.
        sdpn_views_trim (int): Number of samples dropped from the end of each row
            before the Mel transform when `sdpn_views` is enabled.
        sdpn_spec_aug (bool): Whether to erase a band of frames and a band of Mel bins
            from the local views, as SDPN's dataset did until it was commented out on
            2026-06-06 (after the `exp/sdpn` run that the published results come from).
            One band is drawn per sample and erased from both of its local rows, before
            the log, so the erased bins reach the encoder as `log(1e-6)`.
        sdpn_spec_aug_prob (float): Probability of erasing anything for a given sample.
        sdpn_spec_aug_max_time (int): Exclusive upper bound on the number of frames erased.
        sdpn_spec_aug_max_freq (int): Exclusive upper bound on the number of Mel bins erased.
        clip_grad (float): Maximum per-parameter gradient norm applied to the student
            (encoder and head). Defaults to 0, which disables clipping.
    """

    global_count: int = 1
    local_count: int = 4

    start_tau: float = 0.996

    head_hidden_dim: int = 2048
    head_output_dim: int = 256

    nb_prototypes: int = 1024

    student_temperature: float = 0.1
    teacher_temperature: float = 0.025

    memax_weight: float = 1.0
    koleo_weight: float = 0.1

    shared_prototypes: bool = False
    prototypes_grad_world_size_sum: bool = False
    head_activation: str = "relu"
    sdpn_init: bool = False
    prototypes_weight_decay: bool = True
    sdpn_views: bool = False
    sdpn_views_trim: int = 100
    sdpn_spec_aug: bool = False
    sdpn_spec_aug_prob: float = 0.2
    sdpn_spec_aug_max_time: int = 10
    sdpn_spec_aug_max_freq: int = 6
    clip_grad: float = 0.0


class DINOSK(BaseMomentumMethod):
    """
    DINOSK method.

    Attributes:
        head (DINOSKHead): Head module.
        head_momentum (DINOSKHead): Head momentum module.
        loss_fn (DINOSKLoss): Loss function.
    """

    def __init__(
        self,
        config: DINOSKConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a DINOSK method.

        Args:
            config (DINOSKConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.SSPS_NB_POS_EMBEDDINGS = config.global_count

        self.embeddings_dim = config.nb_prototypes

        self.head = DINOSKHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
            nb_prototypes=config.nb_prototypes,
            activation=config.head_activation,
            sdpn_init=config.sdpn_init,
            include_prototypes=not config.shared_prototypes,
        )

        self.head_momentum = DINOSKHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
            nb_prototypes=config.nb_prototypes,
            activation=config.head_activation,
            sdpn_init=config.sdpn_init,
            include_prototypes=not config.shared_prototypes,
        )
        initialize_momentum_params(self.head, self.head_momentum)

        # SDPN holds a single prototype bank, trained by SGD and L2-normalized at
        # use time, which both branches score against. The teacher therefore reads
        # the live prototypes instead of an EMA copy lagging by tau.
        self.prototypes = None
        if config.shared_prototypes:
            sqrt_k = (1.0 / config.head_output_dim) ** 0.5
            self.prototypes = nn.Parameter(
                torch.empty(config.nb_prototypes, config.head_output_dim).uniform_(
                    -sqrt_k, sqrt_k
                )
            )

        # SDPN Mel-transforms each pair of local frames as a single row and only
        # splits them apart afterwards, so the front-end has to sit here rather than
        # inside the encoders (which must run with `extract_mel_features: false`).
        self.features_extractor = None
        if config.sdpn_views:
            enc_config = self.encoder.config
            assert not enc_config.extract_mel_features, (
                "`sdpn_views` requires `extract_mel_features: false` on the encoder"
            )
            self.features_extractor = MelSpectrogram(
                n_fft=enc_config.mel_n_fft,
                win_length=enc_config.mel_win_length,
                hop_length=enc_config.mel_hop_length,
                window_fn=BaseEncoder.MEL_WIN_FUNCTIONS[enc_config.mel_win_fn],
                n_mels=enc_config.mel_n_mels,
                sample_rate=enc_config.mel_sample_rate,
            )
            self.instance_norm = nn.InstanceNorm1d(enc_config.mel_n_mels)

        self.loss_fn = DINOSKLoss(
            global_count=config.global_count,
            local_count=config.local_count,
            student_temp=config.student_temperature,
            teacher_temp=config.teacher_temperature,
            memax_weight=config.memax_weight,
            koleo_weight=config.koleo_weight,
        )

    def extract_mel(self, X: T) -> T:
        """
        Compute the Mel Spectrogram, as SDPN's dataset does.

        Args:
            X (T): Input waveform tensor. Shape: (N, L).

        Returns:
            T: Output tensor. Shape: (N, M, T).
        """
        with torch.no_grad():
            scale = self.encoder.config.waveform_scale
            if self.training and scale != 1.0:
                X = X * scale
            return self.features_extractor(X)

    def normalize_features(self, Z: T) -> T:
        """
        Take the log of Mel Spectrogram features and normalize them, as the encoders
        do internally.

        SDPN runs this on the views themselves, after a pair has been split apart, so
        each view is normalized over its own frames. The two halves of a pair carry
        independent augmentation profiles, hence different statistics.

        Args:
            Z (T): Input tensor. Shape: (N, M, T).

        Returns:
            T: Output tensor. Shape: (N, M, T).
        """
        with torch.no_grad():
            Z = Z + 1e-6
            Z = Z.log()
            return self.instance_norm(Z)

    def extract_features(self, X: T) -> T:
        """
        Extract Mel Spectrogram features, replicating what the encoders do internally.

        Args:
            X (T): Input waveform tensor. Shape: (N, L).

        Returns:
            T: Output tensor. Shape: (N, M, T).
        """
        return self.normalize_features(self.extract_mel(X))

    def apply_sdpn_spec_aug(self, pairs: T, nb_samples: int) -> T:
        """
        Erase a band of frames and a band of Mel bins from the local views.

        SDPN draws one band per sample and erases it from both of that sample's local
        rows, before the pair is split and before the log, so the erased bins reach the
        encoder as `log(1e-6)` rather than as zeros.

        Args:
            pairs (T): Mel Spectrogram of the local pairs. Shape: (P * N, M, F).
            nb_samples (int): Number of samples in the batch (N).

        Returns:
            T: Output tensor. Shape: (P * N, M, F).
        """
        nb_pairs = pairs.size(0) // nb_samples
        _, nb_mels, nb_frames = pairs.shape

        max_time = self.config.sdpn_spec_aug_max_time
        max_freq = self.config.sdpn_spec_aug_max_freq

        for i in range(nb_samples):
            if np.random.random() <= 1 - self.config.sdpn_spec_aug_prob:
                continue

            nb_erased_frames = np.random.randint(0, max_time)
            nb_erased_mels = np.random.randint(0, max_freq)
            start_frame = np.random.randint(0, nb_frames - max_time)
            start_mel = np.random.randint(0, nb_mels - max_freq)

            rows = [i + p * nb_samples for p in range(nb_pairs)]
            pairs[rows, :, start_frame : start_frame + nb_erased_frames] = 0
            pairs[rows, start_mel : start_mel + nb_erased_mels, :] = 0

        return pairs

    def assemble_sdpn_views(self, X: T, L: int) -> Tuple[T, T]:
        """
        Build the global and local features the way SDPN does.

        Local frames arrive concatenated by pairs, every row is trimmed so that the
        Mel transform yields an even number of frames, and each pair is only split in
        half afterwards. Frames sitting on a join therefore see the neighbouring crop
        instead of reflection padding, exactly as in SDPN.

        Args:
            X (T): Input waveform tensor. Shape: (V, N, L).
            L (int): Length of a row of the input tensor.

        Returns:
            Tuple[T, T]: Global and local features.
        """
        nb_global = self.config.global_count

        trim = L - self.config.sdpn_views_trim
        global_frames = self.extract_features(X[:nb_global].reshape(-1, L)[:, :trim])

        # The Mel transform runs on the joined pair, so frames sitting on the join see
        # the neighbouring crop, but the log and the normalization only run once the
        # pair has been split into views.
        pairs = self.extract_mel(X[nb_global:].reshape(-1, L)[:, :trim])

        half = pairs.size(-1) // 2
        nb_pairs = X.size(0) - nb_global

        if self.config.sdpn_spec_aug and self.training:
            pairs = self.apply_sdpn_spec_aug(pairs, X.size(1))
        local_frames = torch.cat(
            [
                half_pair
                for pair in pairs.chunk(nb_pairs)
                for half_pair in (pair[..., :half], pair[..., half:])
            ],
            dim=0,
        )

        return global_frames, self.normalize_features(local_frames)

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T, T, T]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T, T, T]]: Encoder output for inference or embeddings for training.
        """
        if not training:
            if self.features_extractor is not None:
                X = self.extract_features(X)
            return self.encoder_momentum(X)

        N, V, L = X.size()

        X = X.transpose(0, 1)

        if self.features_extractor is not None:
            global_frames, local_frames = self.assemble_sdpn_views(X, L)
        else:
            nb_global = self.config.global_count
            global_frames = X[:nb_global, :, :].reshape(-1, L)
            local = X[nb_global:, :, :]
            if local.size(0) * 2 == self.config.local_count:
                # The dataset concatenated the local frames by pairs, which keeps every
                # row the length of a global frame and avoids padding them. Split them
                # back apart here; unlike `sdpn_views` the Mel is then taken per view.
                local = local.reshape(-1, L)
                local_frames = torch.cat(
                    (local[:, : L // 2], local[:, L // 2 :]), dim=0
                )
            else:
                local_frames = local[:, :, : L // 2].reshape(-1, L // 2)

        Y_global = self.encoder_momentum(global_frames)
        T = self.head_momentum(Y_global)

        Y = self.encoder(local_frames)
        S = self.head(Y)

        if self.prototypes is not None:
            P = F.normalize(self.prototypes, p=2, dim=-1)
            S = S @ P.T
            # SDPN computes the targets under no_grad: the teacher branch never
            # backpropagates into the shared prototypes.
            with torch.no_grad():
                T = T @ P.T

        # Global frames are neither augmented nor cropped and are thus directly
        # used as reference representations for SSPS (no extra frame required).
        Y_ref = None
        if self.ssps:
            Y_ref = F.normalize(Y_global[:N].detach(), p=2, dim=-1)

        return S, T, Y, Y_ref

    def update_optim(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        init_wd: float,
        step: int,
        nb_steps: int,
        nb_steps_per_epoch: int,
    ) -> Tuple[float, float]:
        """
        Update the learning rate for DINOSK method.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            init_lr (float): Initial learning rate from configuration.
            init_wd (float): Initial weight decay from configuration.
            step (int): Current training step.
            nb_steps (int): Total number of training steps.
            nb_steps_per_epoch (int): Number of training steps per epoch.

        Returns:
            Tuple[float, float]: Updated learning rate and initial weight decay.
        """
        lr, wd = super().update_optim(
            optimizer,
            init_lr,
            init_wd,
            step,
            nb_steps,
            nb_steps_per_epoch,
        )

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr
            param_group["weight_decay"] = init_wd if i == 0 else 0

        return lr, init_wd

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = [{"params": self.head.parameters()}]
        if self.prototypes is not None:
            extra_learnable_params.append({"params": [self.prototypes]})
        params = super().get_learnable_params() + extra_learnable_params

        prototypes_ids = {id(p) for p in self.get_prototypes_params()}

        # Do not apply weight decay on biases and norms parameters
        regularized = []
        not_regularized = []
        for module in params:
            for param in module["params"]:
                if not param.requires_grad:
                    continue

                skip_wd = not self.config.prototypes_weight_decay and (
                    id(param) in prototypes_ids
                )

                if len(param.shape) == 1 or skip_wd:
                    not_regularized.append(param)
                else:
                    regularized.append(param)

        return [
            {"params": regularized},
            {"params": not_regularized},
        ]

    def get_prototypes_params(self) -> List[nn.Parameter]:
        """
        Get the learnable prototypes parameters of the student branch.

        Returns:
            List[nn.Parameter]: Collection of parameters.
        """
        if self.prototypes is not None:
            return [self.prototypes]
        return [p for p in self.head.prototypes.parameters() if p.requires_grad]

    def on_after_backward(self):
        """
        Rescale the prototypes gradients and clip the student gradients.

        DDP averages gradients across processes whereas SDPN keeps the prototypes
        outside of the DDP module and sums their gradients (`AllReduceSum`), which
        gives them an effective learning rate `world_size` times larger than the
        rest of the model.

        Clipping is applied afterwards, per parameter rather than on the global norm,
        and to the student only, as SDPN's `clip_gradients` does.

        Returns:
            None
        """
        world_size = get_world_size()
        if self.config.prototypes_grad_world_size_sum and world_size > 1:
            for param in self.get_prototypes_params():
                if param.grad is not None:
                    param.grad.mul_(world_size)

        if self.config.clip_grad:
            student = list(self.encoder.parameters()) + list(self.head.parameters())
            for param in student:
                if param.grad is None:
                    continue
                clip_coef = self.config.clip_grad / (param.grad.norm(2) + 1e-6)
                if clip_coef < 1:
                    param.grad.mul_(clip_coef)

    def get_momentum_pairs(self) -> List[Tuple[nn.Module, nn.Module]]:
        """
        Get a list of modules and their associated momentum module.

        Returns:
            List[Tuple[nn.Module, nn.Module]]: List of (module, module_momentum) pairs.
        """
        extra_momentum_pairs = [(self.head, self.head_momentum)]
        return super().get_momentum_pairs() + extra_momentum_pairs

    def train_step(
        self,
        Z: Tuple[T, T, T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T, T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        S, T, Y, Y_ref = Z

        if self.ssps:
            self.ssps.sample(indices, Y_ref)
            T_views = T.chunk(self.config.global_count)
            T_pp = torch.cat(
                [self.ssps.apply(i, T_i) for i, T_i in enumerate(T_views)]
            )
            self.ssps.update_buffers(step_rel, indices, Y_ref, T_views)
            loss, loss_metrics = self.loss_fn(S, T_pp, Y)
        else:
            loss, loss_metrics = self.loss_fn(S, T, Y)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/tau": self.momentum_updater.tau,
                **loss_metrics
            },
        )

        return loss