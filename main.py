import torch
from pytorch_lightning.cli import LightningCLI

from lightning import AutoregressiveLM, DiffusionLM


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()