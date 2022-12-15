import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from lightning import AutoregressiveLM, DiffusionLM
from lightning.mock import MockData
from lightning.data import ConcatData


def cli_main():
    cli = LightningCLI(
        trainer_defaults={
            'accelerator': 'gpu',
            'strategy': DDPStrategy(find_unused_parameters=False),
            'log_every_n_steps': 1,
            'callbacks': [
                ModelCheckpoint(
                    save_top_k=1,
                    save_last=True,
                    every_n_train_steps=10000,
                    filename='{epoch}-{step}',
                ),
                ModelSummary(max_depth=4)
            ]
        }
    )


if __name__ == "__main__":
    cli_main()
