import pytest

import torch
from torch.optim import Adam

from sslsv.utils.helpers import load_config, load_model
from tests.utils import add_dummy_trainer


def test_basic():
    config = load_config("tests/resources/simple/config.yml", verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)
    model = torch.nn.DataParallel(model)

    add_dummy_trainer(model.module)

    optimizer = Adam(
        model.module.get_learnable_params(),
        lr=0,
        weight_decay=0,
    )

    X1 = torch.randn(256, 2, 32000).cuda()
    X2 = torch.randn(256, 32000).cuda()

    Z = model(X1, training=True)
    loss = model.module.train_step(Z, step=0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    Z = model(X2)

    assert pytest.approx(Z.sum().item()) == 2420.7763671875
