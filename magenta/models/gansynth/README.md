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

GANSynth can train on the NSynth dataset in ~3-4 days on a single V100 GPU. To train, first [follow the setup instructions for Magenta](https://github.com/tensorflow/magenta/blob/master/README.md), using the install or develop environment.

Next, you'll need to access the GANSynth subset of the NSynth Dataset, which has different splits than the original dataset and some additional filtering. There are two ways to do this:

1. Set 'tfds_data_dir' to 'gs://tfds-data/datasets' as shown in the example below. This will read the data directly off of Google Cloud Storage and is recommended if you're running your training on a Google Cloud VM or in Google Colab. Your training may be bottlenecked by I/O if you're not training in one of these places.
1. Copy the dataset locally, which will remove a potential I/O bottleneck during training. You can download the dataset by running the following command with your own local dir:

```bash
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=nsynth/gansynth_subset --tfds_dir=/path/to/local/dir
```

To test that training works, run from the following command, replacing 'gs://tfds-data/datasets' with your local directory if you used option 2 above:

```bash
gansynth_train.py --hparams='{"tfds_data_dir":"gs://tfds-data/datasets", "train_root_dir":"/tmp/gansynth/train"}'
```

This will run the model with suitable hyperparmeters for quickly testing training (which you can find in `model.py`). The best performing hyperparmeter configuration from the paper _(Mel-Spectrograms, Progressive Training, High Frequency Resolution)_, can be found in `configs/mel_prog_hires.py`. You can train with this config by adding it as a flag:

```bash
gansynth_train --config=mel_prog_hires --hparams='{"tfds_data_dir":"gs://tfds-data/datasets" "train_root_dir":"/tmp/gansynth/train"}'
```
