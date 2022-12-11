import argparse
import torch
import soundfile as sf
import yaml
from importlib import import_module
import note_seq
from preprocessor.event_codec import Codec
from preprocessor.preprocessor import preprocess
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf


@torch.no_grad()
def diff_main(model, tokens, segment_length, spec_frames, with_context, spec2wav):
    output_specs = []
    zero_wav_context = torch.zeros(
        1, segment_length).cuda() if with_context else None
    mel_context = None
    for x in tqdm(tokens):
        x = x.unsqueeze(0).cuda()
        if len(output_specs):
            pred = model(x, seq_length=spec_frames,
                         mel_context=mel_context, rescale=False)
        else:
            pred = model(x, seq_length=spec_frames,
                         wav_context=zero_wav_context, rescale=False)

        output_specs.append(pred)
        mel_context = pred if with_context else None

    output_specs = torch.cat(output_specs, dim=1)
    output_specs = model.mel[1].reverse(output_specs)
    output_specs = tf.convert_to_tensor(output_specs.cpu().numpy())
    pred_wav = spec2wav(output_specs)
    return pred_wav.numpy().flatten()


@torch.no_grad()
def ar_main(model, tokens, spec_frames, dither_amount, spec2wav):
    output_specs = []
    for x in tqdm(tokens):
        x = x.unsqueeze(0).cuda()
        pred = model(x, max_len=spec_frames, dither_amount=dither_amount)
        output_specs.append(pred)

    output_specs = torch.cat(output_specs, dim=1)
    output_specs = tf.convert_to_tensor(output_specs.cpu().numpy())
    pred_wav = spec2wav(output_specs)
    return pred_wav.numpy().flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('midi', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('config', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-W', type=float, default=None)
    parser.add_argument('--dither', type=float, default=0.0)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_configs = config['model']

    if args.W is not None:
        model_configs['init_args']['cfg_weighting'] = args.W

    module_path, class_name = model_configs['class_path'].rsplit('.', 1)
    module = import_module(module_path)
    model = getattr(module, class_name).load_from_checkpoint(
        args.ckpt, **model_configs['init_args'])
    model = model.cuda()
    model.eval()

    hop_length = model_configs['init_args']['hop_length']
    n_mels = model_configs['init_args']['n_mels']
    data_configs = config['data']
    sr = data_configs['init_args']['sample_rate']
    segment_length = data_configs['init_args']['segment_length']
    spec_frames = segment_length // hop_length
    resolution = 100
    segment_length_in_time = segment_length / sr
    codec = Codec(int(segment_length_in_time * resolution + 1))

    with_context = data_configs['init_args']['with_context'] and model_configs['init_args']['with_context']

    ns = note_seq.midi_file_to_note_sequence(args.midi)
    ns = note_seq.apply_sustain_control_changes(ns)
    tokens, _ = preprocess(ns, codec=codec)

    spec2wav = hub.KerasLayer(
        'https://tfhub.dev/google/soundstream/mel/decoder/music/1')

    if class_name == 'AutoregressiveLM':
        pred = ar_main(model, tokens, spec_frames,
                       args.dither, spec2wav)
    else:
        pred = diff_main(model, tokens, segment_length,
                         spec_frames, with_context, spec2wav)
    sf.write(args.output, pred, sr)
