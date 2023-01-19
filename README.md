# Multi-instrument Music Synthesis with Spectrogram Diffusion

An unofficial implementation of the paper [Multi-instrument Music Synthesis with Spectrogram Diffusion](https://arxiv.org/abs/2206.05408), adapted from [official codebase](https://github.com/magenta/music-spectrogram-diffusion).

## Data Preparation

Please download the following datasets.

* [MusicNet](https://doi.org/10.5281/zenodo.5120004)
* [Maestro](https://magenta.tensorflow.org/datasets/maestro#dataset)
* [GuitarSet](https://guitarset.weebly.com/)
* [URMP](https://labsites.rochester.edu/air/projects/URMP.html)
* [Slakh2100](https://doi.org/10.5281/zenodo.4599666)

### Create Clean MIDI for URMP

## Training

### Small Autoregressive

```
python main.py fit --config cfg/ar_small.yaml
```

### Diffusion, Small Without Context

```
python main.py fit --config cfg/diff_small.yaml
```

### Diffusion, Small With Context

```
python main.py fit --config cfg/diff_small.yaml --data.init_args.with_context true --model.init_args.with_context true
```

### Diffusion, Base With Context

```
python main.py fit --config cfg/diff_base.yaml
```

Remember to change the path arguments under the `data` section of the yaml files to where you downloaded the dataset, or set it using `--data.init_args.*_path` keyword in commandline.
You can also set the path to `null` if you want to ommit that dataset.

To adjust other hyperparmeters, please refer to [LightningCLI documentation](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html) for more information.

## Evaluating

## Inferencing

## Checkpoints


## TODO

- [ ] Use [MidiTok](https://github.com/Natooz/MidiTok) for tokenization.
- [ ] Use [torchvggish](https://github.com/harritaylor/torchvggish) for vggish embeddings.
- [ ] Remove context encoder and use inpainting techniques for segment-by-segment generation, similar to https://github.com/archinetai/audio-diffusion-pytorch.
- [ ] Encoder-free Classifier-Guidance generation with [MT3](https://github.com/magenta/mt3).