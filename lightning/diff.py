import pytorch_lightning as pl
import torch_optimizer as optim
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from models.diff_decoder import MIDI2SpecDiff
from .mel import MelFeature
from .scaler import get_scaler


def log_snr2snr(log_snr: Tensor) -> Tensor:
    return torch.exp(log_snr)


def snr2as(snr: Tensor):
    snr_p1 = snr + 1
    return torch.sqrt(snr / snr_p1), snr_p1.reciprocal()


def log_snr2as(log_snr: Tensor):
    var = (-log_snr).sigmoid()
    return (1 - var).sqrt(), var


def log_snr2logas(log_snr: Tensor):
    log_var = -F.softplus(log_snr)
    return 0.5 * (log_snr + log_var), log_var


class DiffusionLM(pl.LightningModule):
    logsnr_min = -20
    logsnr_max = 20

    def __init__(self,
                 num_emb: int = 900,
                 output_dim: int = 128,
                 max_input_length: int = 2048,
                 max_output_length: int = 512,
                 emb_dim: int = 512,
                 nhead: int = 6,
                 head_dim: int = 64,
                 num_layers: int = 8,
                 cfg_dropout: float = 0.1,
                 cfg_weighting: float = 2.0,
                 with_context: bool = False,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = True,
                 **mel_kwargs) -> None:
        super().__init__()

        self.cfg_dropout = cfg_dropout
        self.cfg_weighting = cfg_weighting
        self.output_dim = output_dim

        self.model = MIDI2SpecDiff(
            num_emb=num_emb, output_dim=output_dim, max_input_length=max_input_length,
            max_output_length=max_output_length, emb_dim=emb_dim, nhead=nhead, with_context=with_context,
            head_dim=head_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout,
            layer_norm_eps=layer_norm_eps, norm_first=norm_first,
        )

        self.mel = nn.Sequential(
            MelFeature(window_fn=torch.hann_window, **mel_kwargs),
            get_scaler()
        )

    def get_log_snr(self, t):
        """Compute Cosine log SNR for a given time step."""
        b = math.atan(math.exp(-0.5 * self.logsnr_max))
        a = math.atan(math.exp(-0.5 * self.logsnr_min)) - b
        return -2.0 * torch.log(torch.tan(a * t + b))

    def get_training_inputs(self, x: torch.Tensor, uniform: bool = False):
        N = x.shape[0]
        if uniform:
            t = torch.linspace(0, 1, N).to(x.device)
        else:
            t = x.new_empty(N).uniform_(0, 1)
        log_snr = self.get_log_snr(t)
        alpha, var = log_snr2as(log_snr)
        sigma = var.sqrt()
        noise = torch.randn_like(x)
        z_t = x * alpha[:, None, None] + sigma[:, None, None] * noise
        return z_t, t, noise

    def forward(self, midi: Tensor, seq_length=512, context=None, T=1000):
        t = torch.linspace(0, 1, T).to(midi.device)
        log_snr = self.get_log_snr(t)
        log_alpha, log_var = log_snr2logas(log_snr)

        var = log_var.exp()
        alpha = log_alpha.exp()
        alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
        c = -torch.expm1(log_snr[1:] - log_snr[:-1])
        c.relu_()

        z_t = midi.new_empty(
            midi.shape[0], seq_length, self.output_dim).normal_()

        dropout_mask = torch.tensor(
            [0] * midi.shape[0] + [1] * midi.shape[0]).bool().to(midi.device)
        t = torch.broadcast_to(t, (midi.shape[0] * 2, T))
        midi = midi.repeat(2, 1)
        if context is not None:
            context = context.repeat(2, 1, 1)

        for t_idx in tqdm(range(T - 1, -1, -1)):
            s_idx = t_idx - 1
            noise_hat = self.model(midi, z_t.repeat(
                2, 1, 1), t[:, t_idx], context, dropout_mask=dropout_mask)
            cond_noise_hat, uncond_noise_hat = noise_hat.chunk(2, dim=0)
            noise_hat = cond_noise_hat * self.cfg_weighting + \
                uncond_noise_hat * (1 - self.cfg_weighting)
            if s_idx >= 0:
                mu = (z_t - var[t_idx].sqrt() * c[s_idx]
                      * noise_hat) * alpha_st[s_idx]
                z_t = mu + (var[s_idx] * c[s_idx]).sqrt() * \
                    torch.randn_like(z_t)
                continue
            final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]
        return final

    def training_step(self, batch, batch_idx):
        midi, wav, *_ = batch
        spec = self.mel(wav)
        if len(_) > 0:
            context = _[0]
            context = self.mel(context)
        else:
            context = None
        N = midi.shape[0]
        dropout_mask = spec.new_empty(N).bernoulli_(self.cfg_dropout).bool()
        z_t, t, noise = self.get_training_inputs(spec)
        noise_hat = self.model(midi, z_t, t, context,
                               dropout_mask=dropout_mask)
        loss = F.l1_loss(noise_hat, noise)

        values = {
            'loss': loss,
        }
        self.log_dict(values, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        midi, wav, *_ = batch
        spec = self.mel(wav)
        if len(_) > 0:
            context = _[0]
            context = self.mel(context)
        else:
            context = None
        z_t, t, noise = self.get_training_inputs(spec, uniform=True)
        noise_hat = self.model(midi, z_t, t, context)
        loss = F.l1_loss(noise_hat, noise)

        values = {
            'val_loss': loss,
        }
        self.log_dict(values, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return optim.Adafactor(self.parameters(), lr=1e-3)
