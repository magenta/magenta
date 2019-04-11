# GANSynth

GANSynth is an algorithm for synthesizing audio with generative adversarial networks.
The details can be found in the [ICLR 2019 Paper](https://openreview.net/forum?id=H1xQVn09FX). It achieves better audio quality than a standard WaveNet baselines on the [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth), and synthesizes audio thousands of times faster.

## Generation

To generate some sounds, first [follow the setup instructions for Magenta](https://github.com/tensorflow/magenta/blob/master/README.md), then download a pretrained checkpoint, or train your own. We have several available for download:

* [acoustic_only](https://storage.googleapis.com/magentadata/models/gansynth/acoustic_only.zip): As shown in the paper, trained on only acoustic instruments pitch 24-84 (Mel-IF, Progressive, High Frequency Resolution).

* [all_instruments](https://storage.googleapis.com/magentadata/models/gansynth/all_instruments.zip): Trained on all instruments pitch 24-84 (Mel-IF, Progressive, High Frequency Resolution).

You can start by generating some random sounds (random pitch and latent vector) by unzipping the checkpoint and running the generate script from the root of the Magenta directory.

```bash
python magenta/models/gansynth/gansynth_generate.py --ckpt_dir=/path/to/acoustic_only --output_dir=/path/to/output/dir --midi_file=/path/to/file.mid
```

If a MIDI file is specified, notes are synthesized with interpolation between latent vectors in time. If no MIDI file is given, a random batch of notes is synthesized.

If you've installed from the pip package, it will install a console script so you can run from anywhere.
```bash
gansynth_generate --ckpt_dir=/path/to/acoustic_only --output_dir=/path/to/output/dir --midi_file=/path/to/file.mid
```


## Training

GANSynth can train on the NSynth dataset in ~3-4 days on a single V100 GPU. To train, first [follow the setup instructions for Magenta](https://github.com/tensorflow/magenta/blob/master/README.md), using the install or develop environment. Then download the [NSynth Datasets](https://magenta.tensorflow.org/datasets/nsynth) as TFRecords.

To test that training works, run from the root of the Magenta repo directory:

```bash
python magenta/models/gansynth/gansynth_train.py --hparams='{"train_data_path":"/path/to/nsynth-train.tfrecord", "train_root_dir":"/tmp/gansynth/train"}'
```

This will run the model with suitable hyperparmeters for quickly testing training (which you can find in `model.py`). The best performing hyperparmeter configuration from the paper _(Mel-Spectrograms, Progressive Training, High Frequency Resolution)_, can be found in `configs/mel_prog_hires.py`. You can train with this config by adding it as a flag:

```bash
python magenta/models/gansynth/gansynth_train.py --config=mel_prog_hires --hparams='{"train_data_path":"/path/to/nsynth-train.tfrecord", "train_root_dir":"/tmp/gansynth/train"}'
```

You can also alter it or make other configs to explore the other representations. As a reminder, the full list of hyperparameters can be found in `model.py`. By default, the model trains only on acoustic instruments pitch 24-84 as in the paper. This can be changed in `datasets.py`.

If you've installed from the pip package, it will install a console script so you can run from anywhere.
```bash
gansynth_train --config=mel_prog_hires --hparams='{"train_data_path":"/path/to/nsynth-train.tfrecord", "train_root_dir":"/tmp/gansynth/train"}'
```

