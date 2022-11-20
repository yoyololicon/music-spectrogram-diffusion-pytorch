import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from lightning import AutoregressiveLM, DiffusionLM
from lightning.mock import MockData


def cli_main():
    cli = LightningCLI(
        trainer_defaults={
            'accelerator': 'gpu',
            'strategy': 'ddp',
            'log_every_n_steps': 1,
            'callbacks': [
                ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    save_last=True,
                    every_n_epochs=1,
                    filename='{epoch}-{step}-{val_loss:.2f}',
                ),
                ModelSummary(max_depth=4)
            ]
        }
    )


if __name__ == "__main__":
    cli_main()
