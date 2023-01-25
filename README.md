# Multi-instrument Music Synthesis with Spectrogram Diffusion

An unofficial PyTorch implementation of the paper [Multi-instrument Music Synthesis with Spectrogram Diffusion](https://arxiv.org/abs/2206.05408), adapted from [official codebase](https://github.com/magenta/music-spectrogram-diffusion).
We aim to increase the reproducibility of their work by providing training code and pre-trained models in PyTorch.

## Data Preparation

Please download the following datasets.

* [MusicNet](https://doi.org/10.5281/zenodo.5120004)
* [Maestro](https://magenta.tensorflow.org/datasets/maestro#dataset)
* [GuitarSet](https://guitarset.weebly.com/)
* [URMP](https://labsites.rochester.edu/air/projects/URMP.html)
* [Slakh2100](https://doi.org/10.5281/zenodo.4599666)

### Create Clean MIDI for URMP

The MIDI files in the URMP dataset mostly don't contain the correct program number. Use the `clean_urmp_midi.py` script to create a new set of MIDI files that contain the correct program number corresponding to the instruments in the file names.

## Training

### Small Autoregressive

```
python main.py fit --config cfg/ar_small.yaml
```

### Diffusion, Small without Context

```
python main.py fit --config cfg/diff_small.yaml
```

### Diffusion, Small with Context

```
python main.py fit --config cfg/diff_small.yaml --data.init_args.with_context true --model.init_args.with_context true
```

### Diffusion, Base with Context

```
python main.py fit --config cfg/diff_base.yaml
```

Remember to change the path arguments under the `data` section of the yaml files to where you downloaded the dataset, or set them using `--data.init_args.*_path` keyword in commandline.
You can also set the path to `null` if you want to ommit that dataset.
Notice that URMP requires one extra path argument, which is where you create [the clean MIDI](#create-clean-midi-for-urmp).

To adjust other hyperparmeters, please refer to [LightningCLI documentation](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html) for more information.

## Evaluating

The following command will compute the Reconstruction and FAD metrics using the embeddings from the VGGish and TRILL models and reporting the averages across the whole test dataset.

```
python main.py test --config config.yaml --ckpt_path your_checkpoint.ckpt
```

## Inferencing

To synthesize audio from MIDI with trained models:

```
python infer.py input.mid checkpoint.ckpt config.yaml output.wav
```

## Pre-Trained Models

We provided three pre-trained models corresponding to the diffusion baselines in the paper.
* [small without context](https://drive.google.com/file/d/1rw6xNMgtIOVHKfXgqzYseo_-Qihkp9NR/view?usp=share_link)
* [small with context](https://drive.google.com/file/d/1JS-u2dj2p6rOsUEEuhYh_OnjodNsBo0k/view?usp=share_link)
* [base with context](https://drive.google.com/file/d/1CO1WGjHMNvsr5OoAdBKKACNDRinZkwVW/view?usp=share_link)
* [generated samples](https://drive.google.com/drive/folders/1UO04q-oAbL1shNaztcCNnO-MNATE6tWa?usp=share_link)

We trained them following the settings in the paper besides the batch size, which we reduced to 8 due to limited computational resources.
We evaluated these models using our codebase and summarized them in the following table:

|        Models        | VGGish Recon | VGGish FAD | Trill Recon  | Trill FAD |
|:--------------------:|:------------:|:----------:|:------------:|:---------:|
|   Small w/o Context  |     2.48     |    0.49    |     0.84     |    0.08   |
|   Small w/ Context   |     2.44     |    0.59    |     0.68     |    0.04   |
|    Base w/ Context   |       -      |      -     |       -      |     -     |
| Ground Truth Encoded |     1.80     |    0.80    |     0.35     |    0.02   |



## TODO

- [ ] Use [MidiTok](https://github.com/Natooz/MidiTok) for tokenization.
- [ ] Use [torchvggish](https://github.com/harritaylor/torchvggish) for vggish embeddings.
- [ ] Remove context encoder and use inpainting techniques for segment-by-segment generation, similar to https://github.com/archinetai/audio-diffusion-pytorch.
- [ ] Encoder-free Classifier-Guidance generation with [MT3](https://github.com/magenta/mt3).