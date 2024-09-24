import torch

from pathlib import Path

from sslsv.trainer.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_train_dataloader, load_model, evaluate


def test_basic():
    config = load_config("tests/resources/simple/config.yml")

    (config.model_ckpt_path / "model_latest.pt").unlink(missing_ok=True)
    (config.model_ckpt_path / "model_best.pt").unlink(missing_ok=True)

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
