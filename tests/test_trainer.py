import torch

from pathlib import Path
import shutil

from sslsv.trainer.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_train_dataloader, load_model, evaluate

# import pytest

# pytest.skip("Skipping this file temporarily", allow_module_level=True)


def test_basic():
    config = load_config("tests/resources/simple/config.yml")

    if config.model_ckpt_path.exists() and config.model_ckpt_path.is_dir():
        shutil.rmtree(config.model_ckpt_path)

    train_dataloader = load_train_dataloader(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)
    model = torch.nn.DataParallel(model)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        evaluate=evaluate,
        device=device,
    )
    trainer.start()
