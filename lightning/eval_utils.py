import numpy as np
from scipy import linalg
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
import librosa

MODEL_SAMPLE_RATE = 16000

def _resample_and_pad(data):
    data = np.nan_to_num(data)
    #data = librosa.resample(data, 16000, MODEL_SAMPLE_RATE)
    length = len(data)
    target_length = int(np.ceil(length / MODEL_SAMPLE_RATE)
                        ) * MODEL_SAMPLE_RATE
    padding = target_length - length
    data = np.pad(
        data, (padding // 2, padding - padding // 2), mode="constant"
    )
    return data


def get_models():
    trill_model = hub.load(
        "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3"
    )
    vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
    spec2wav = hub.KerasLayer(
        'https://tfhub.dev/google/soundstream/mel/decoder/music/1')
    return vggish_model, trill_model, spec2wav 


def get_wav(model, spec):
    spec = tf.convert_to_tensor(spec.cpu().numpy().astype(np.float32))
    return model(spec).numpy()


def _get_embedding(data, model_fn):
    embeddings = np.vstack([
        model_fn(d, MODEL_SAMPLE_RATE).numpy() for d in data
    ])
    return embeddings


def _get_frechet_distance(true_mu, true_sigma, pred_mu, pred_sigma, eps=1e-6):
    """
    Get FAD distance between two embedding samples
    Implementation Reference: https://github.com/gudgud96/frechet-audio-distance
    """
    true_mu = np.atleast_1d(true_mu)
    pred_mu = np.atleast_1d(pred_mu)
    true_sigma = np.atleast_2d(true_sigma)
    pred_sigma = np.atleast_2d(pred_sigma)

    assert (
        pred_mu.shape == true_mu.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        pred_sigma.shape == true_sigma.shape
    ), "Training and test covariances have different dimensions"

    diff = pred_mu - true_mu

    covmean, _ = linalg.sqrtm(pred_sigma.dot(true_sigma), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(pred_sigma.shape[0]) * eps
        covmean = linalg.sqrtm((pred_sigma + offset).dot(true_sigma + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(pred_sigma) + np.trace(true_sigma) - 2 * tr_covmean


def calculate_metrics(orig_wav, pred_wav, vggish_fn, trill_fn, true_dist, pred_dist):
    metrics = {}
    for name, fn in [('trill', trill_fn), ('vggish', vggish_fn)]:
        pred_embedding = _get_embedding(pred_wav, fn)
        true_embedding = _get_embedding(orig_wav, fn)
<<<<<<< HEAD
        max_length = min(len(pred_embedding), len(true_embedding))
        # metrics[name] = np.linalg.norm(pred_embedding - true_embedding, axis=1).mean()
        metrics[name] = np.linalg.norm(pred_embedding[:max_length] - true_embedding[:max_length], axis=1).mean()
=======
        metrics[name] = np.linalg.norm(pred_embedding - true_embedding, axis=1).mean()
>>>>>>> master
        pred_dist[name].update(pred_embedding)
        true_dist[name].update(true_embedding)
    return metrics


class StreamingMultivariateGaussian(object):
  """Streaming mean and covariance for multivariate Gaussian.
     Reference: https://github.com/magenta/music-spectrogram-diffusion
  """

  def __init__(self):
    self.n = 0
    self.mu = None
    self._sigma_accum = None

  def update(self, x):
    """Update mean and covariance with new data points."""
    n, _ = x.shape
    if self.n == 0:
      self.n = n
      self.mu = np.mean(x, axis=0)
      x_res = x - self.mu[np.newaxis, :]
      self._sigma_accum = np.dot(x_res.T, x_res)
    else:
      x_res_pre = x - self.mu[np.newaxis, :]
      self.n += n
      self.mu += np.sum(x_res_pre, axis=0) / self.n
      x_res_post = x - self.mu[np.newaxis, :]
      self._sigma_accum += np.dot(x_res_pre.T, x_res_post)

  @property
  def sigma(self):
    return self._sigma_accum / self.n


def aggregate_metrics(metrics, true_dists, pred_dists):
    assert len(metrics) > 0, "Should have at least one segment"
    metric = dict() 
    metric["evaluation loss"] = sum(m["loss"] for m in metrics) / len(metrics)
    for name in ["vggish", "trill"]:
        metric[f"{name}_recon"] = sum(m[name] for m in metrics) / len(metrics)
        metric[f"{name}_fad"]= _get_frechet_distance(
            true_dists[name].mu, 
            true_dists[name].sigma,
            pred_dists[name].mu, 
            pred_dists[name].sigma
        )

    return metric
