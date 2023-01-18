# Multi-instrument Music Synthesis with Spectrogram Diffusion

An unofficial implementation of the paper [Multi-instrument Music Synthesis with Spectrogram Diffusion](https://arxiv.org/abs/2206.05408), adapted from [official codebase](https://github.com/magenta/music-spectrogram-diffusion).



## TODO

- [ ] Use [MidiTok](https://github.com/Natooz/MidiTok) for tokenization.
- [ ] Use [torchvggish](https://github.com/harritaylor/torchvggish) for vggish embeddings.
- [ ] Remove context encoder and use inpainting techniques for segment-by-segment generation, similar to https://github.com/archinetai/audio-diffusion-pytorch.
- [ ] Encoder-free Classifier-Guidance generation with [MT3](https://github.com/magenta/mt3).