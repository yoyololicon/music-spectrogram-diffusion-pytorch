import numpy as np
from scipy import linalg
import tensorflow as tf
import tensorflow_hub as hub

MODEL_SAMPLE_RATE = 16000


def _resample_and_pad(data):
    length = data.shape[-1]
    target_length = int(np.ceil(length / MODEL_SAMPLE_RATE)
                        ) * MODEL_SAMPLE_RATE
    padding = target_length - length
    data = np.pad(
        data, ((0, 0), (padding // 2, padding - padding // 2)), mode="constant"
    )
    return data


def get_models():
    trill_model = hub.load(
        "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3"
    )
    vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
    melgan = hub.load(
        "https://tfhub.dev/google/soundstream/mel/decoder/music/1")
    return vggish_model, trill_model, melgan


def get_wav(model, spec):
    spec = tf.convert_to_tensor(spec.cpu().numpy().astype(np.float32))
    return model(spec).numpy()


def _get_embedding(data, model_fn):
    embeddings = np.vstack(
        [model_fn(d, MODEL_SAMPLE_RATE) for d in data]
    )
    return embeddings


def _get_frechet_distance(true_embeddings, pred_embeddings, eps=1e-6):
    """
    Get FAD distance between two embedding samples
    Implementation Reference: https://github.com/gudgud96/frechet-audio-distance
    """
    true_mu = true_embeddings.mean(axis=0)
    true_sigma = np.cov(true_embeddings, rowvar=False)
    pred_mu = pred_embeddings.mean(axis=0)
    pred_sigma = np.cov(pred_embeddings, rowvar=False)

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


def calculate_metrics(orig_wav, pred_wav, vggish_fn, trill_fn):
    vggish_true_embedding = _get_embedding(orig_wav, vggish_fn)
    vggish_pred_embedding = _get_embedding(pred_wav, vggish_fn)
    trill_true_embedding = _get_embedding(orig_wav, trill_fn)
    trill_pred_embedding = _get_embedding(pred_wav, trill_fn)
    metrics = {
        "vggish_true_embedding": vggish_true_embedding,
        "vggish_pred_embedding": vggish_pred_embedding,
        "trill_true_embedding": trill_true_embedding,
        "trill_pred_embedding": trill_pred_embedding,
    }
    return metrics


def aggregate_metrics(metrics):
    assert len(metrics) > 0, "Should have at least one segment"
    if len(metrics) <= 0:
        print()
    loss = sum(m["loss"] for m in metrics) / len(metrics)

    vggish_true_embedding = np.vstack(
        [m["vggish_true_embedding"] for m in metrics]
    )
    vggish_pred_embedding = np.vstack(
        [m["vggish_pred_embedding"] for m in metrics]
    )
    trill_true_embedding = np.vstack(
        [m["trill_true_embedding"] for m in metrics]
    )
    trill_pred_embedding = np.vstack(
        [m["trill_pred_embedding"] for m in metrics]
    )

    vggish_recon = np.linalg.norm(
        vggish_pred_embedding - vggish_true_embedding, axis=1
    ).mean(axis=0)
    trill_recon = np.linalg.norm(
        trill_pred_embedding - trill_true_embedding, axis=1
    ).mean(axis=0)

    vggish_fad = _get_frechet_distance(
        vggish_true_embedding, vggish_pred_embedding)
    trill_fad = _get_frechet_distance(
        trill_true_embedding, trill_pred_embedding)

    return {
        "Evaluation Loss": loss,
        "VGGish Recon": vggish_recon,
        "VGGish FAD": vggish_fad,
        "Trill Recon": trill_recon,
        "Trill FAD": trill_fad,
    }
