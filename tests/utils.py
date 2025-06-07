from types import SimpleNamespace


def add_dummy_trainer(model):
    model.trainer = SimpleNamespace()
    model.trainer.config = SimpleNamespace()
    model.trainer.config.dataset = SimpleNamespace()
    model.trainer.config.dataset.frame_length = 32000
