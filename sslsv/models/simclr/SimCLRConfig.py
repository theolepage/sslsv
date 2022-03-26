from dataclasses import dataclass, field
from typing import List

from sslsv.configs import ModelConfig


@dataclass
class SimCLRConfig(ModelConfig):
    enable_mlp: bool = True
    mlp_dim: int = 2048
    
    infonce_weight: float = 1.0
    vicreg_weight: float = 0.0
    barlowtwins_weight: float = 0.0

    vic_inv_weight: float = 1.0
    vic_var_weight: float = 1.0
    vic_cov_weight: float = 0.04
    
    barlowtwins_lambda: float = 0.005

    representations_losses: List[bool] = field(
        default_factory=lambda: [False, False, False],
        metadata={'help': 'Losses to apply on representations. Format: (infonce, vicreg, barlowtwins)'}
    )
    embeddings_losses: List[bool] = field(
        default_factory=lambda: [True, True, True],
        metadata={'help': 'Losses to apply to embeddings. Format: (infonce, vicreg, barlowtwins)'}
    )