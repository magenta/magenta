# NSynth: Neural Audio Synthesis

NSynth is a WaveNet-based autoencoder for synthesizing audio.

# Background

[WaveNet][wavenet-blog] is an expressive model for temporal sequences such as
speech and music. As a deep autoregressive network of dilated convolutions, it
models sound one sample at a time, similar to a nonlinear infinite impulse
response filter. Since the context of this filter is currently limited to
several thousand samples (about half a second), long-term structure requires a
guiding external signal. [Prior work][wavenet-paper] demonstrated this in the
case of text-to-speech and used previously learned linguistic embeddings to
create impressive results.

In NSynth, we removed the need for conditioning on external features by
employing a WaveNet-style autoencoder to learn its own temporal embeddings.

A full description of the algorithm and accompanying dataset can be found in our
[arXiv paper][arXiv] and [blog post][blog].

# The Models

This repository contains a baseline spectral autoencoder model and a WaveNet autoencoder model, each in their respective directories. The baseline model uses a spectrogram with fft_size 1024 and hop_size 256, MSE loss on the magnitudes, and the Griffin-Lim algorithm for reconstruction. The WaveNet model trains on mu-law encoded waveform chunks of size 6144. It learns embeddings with 16 dimensions that are downsampled by 512 in time.

Given the difficulty of training, we've included weights of models pretrained on the NSynth dataset. They are available for download as TensorFlow checkpoints:

* [Baseline][baseline-ckpt]
* [WaveNet][wavenet-ckpt]


# Training

To train the model you first need a dataset containing raw audio. We have built
a very large dataset of musical notes that you can use for this purpose:
[the NSynth Dataset][dataset].

Training for both these models is very expensive, and likely difficult for many practical setups. Nevertheless, We've included training code for completeness and transparency. The WaveNet model takes around 10 days on 32 K40 gpus (synchronous) to converge at ~200k iterations. The baseline model takes about 5 days on 6 K40 gpus (asynchronous).

Example Usage:
-------

(Baseline)
```bash
bazel run //magenta/models/nsynth/baseline:train -- \
--train_path=/<path>/nsynth-train.tfrecord \
---logdir=/<path>
```

(WaveNet)
```bash
bazel run //magenta/models/nsynth/wavenet/train -- \
--train_path=/<path>/nsynth-train.tfrecord \
--logdir=/<path>
```

The WaveNet training also requires tensorflow 1.1.0-rc1 or beyond.
As of 04/05/17 this requires installing tensorflow from source
(https://github.com/tensorflow/tensorflow/releases).


# Saving Embeddings

We've included scripts for saving embeddings from your own wave files (16kHz audio expected).

Example Usage:
-------

(Baseline)
```bash
bazel run //magenta/models/nsynth/baseline:save_embeddings -- \
--tfrecord_path=/<path>/nsynth-test.tfrecord \
--checkpoint_path=/<path>/baseline-ckpt/model.ckpt-200000 \
--savedir=/<path> \
```

(WaveNet)
```bash
bazel run //magenta/models/nsynth/wavenet:save_embeddings -- \
--checkpoint_path=/<path>/wavenet-ckpt/model.ckpt-200000 \
--wavdir=/<path> \
--savedir=/<path> \
--sample_length=5120 \
--batch_size=4
```


[arXiv]: https://arxiv.org/abs/1704.01279
[baseline-ckpt]:http://download.magenta.tensorflow.org/models/nsynth/baseline-ckpt.tar
[blog]: https://magenta.tensorflow.org/nsynth
[dataset]: https://magenta.tensorflow.org/datasets/nsynth
[wavenet-blog]:https://deepmind.com/blog/wavenet-generative-model-raw-audio/
[wavenet-paper]:https://arxiv.org/abs/1609.03499
[wavenet-ckpt]:http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar
