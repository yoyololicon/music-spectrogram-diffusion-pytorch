from collections import defaultdict
import pytorch_lightning as pl
import torch
from .mel import MelFeature
from .eval_utils import get_models, calculate_metrics, aggregate_metrics, get_wav, StreamingMultivariateGaussian


class GTEncoded(pl.LightningModule):
    def __init__(self,
                 **mel_kwargs) -> None:
        super().__init__()
        self.mel = MelFeature(window_fn=torch.hann_window, **mel_kwargs)

    def on_test_start(self) -> None:
        vggish_model, trill_model, self.melgan = get_models()
        self.vggish_fn = lambda x, sr: vggish_model(x)
        self.trill_fn = lambda x, sr: trill_model(
            x, sample_rate=sr)['embedding']

        self.true_dists = defaultdict(StreamingMultivariateGaussian)
        self.pred_dists = defaultdict(StreamingMultivariateGaussian)

        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        _, orig_wav, *_ = batch
        spec = self.mel(orig_wav)
        pred_wav = self.spec_to_wav(spec)
        orig_wav = orig_wav.cpu().numpy()
        metric = calculate_metrics(
            orig_wav, pred_wav, self.vggish_fn, self.trill_fn, self.true_dists, self.pred_dists)
        metric['loss'] = 0
        return metric

    def test_epoch_end(self, outputs) -> None:
        metrics = aggregate_metrics(outputs, self.true_dists, self.pred_dists)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return super().test_epoch_end(outputs)

    def spec_to_wav(self, spec):
        return get_wav(self.melgan, spec)
