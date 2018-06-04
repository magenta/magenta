## Onsets and Frames: Dual-Objective Piano Transcription

For model details, see our paper on arXiv:
[Onsets and Frames: Dual-Objective Piano Transcription](https://arxiv.org/abs/1710.11153). You can also listen to the [Audio Examples](http://download.magenta.tensorflow.org/models/onsets_frames_transcription/index.html) described in the paper.

## Colab Notebook

The easiest way to use the model is with our [Onsets and Frames Colab Notebook](https://colab.research.google.com/notebook#fileId=/v2/external/notebooks/magenta/onsets_frames_transcription/onsets_frames_transcription.ipynb). You can upload arbitrary audio files and receive a transcription without installing any software.

## How to Use

If you would like to run the model locally, first set up your [Magenta environment](/README.md). Next, you can either use a pre-trained model or train your own.

## Dataset creation

First, you'll need to download a copy of the
[MAPS Database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/).
Unzip the MAPS zip files after you've downloaded them.

Next, you'll need to create TFRecord files that contain the relevant data from MAPS by running the following command:

```bash
MAPS_DIR=<path to directory containing unzipped MAPS dataset>
OUTPUT_DIR=<path where the output TFRecord files should be stored>

onsets_frames_transcription_create_dataset \
  --input_dir="${MAPS_DIR}" \
  --output_dir="${OUTPUT_DIR}"
```

## Pre-trained

To try inference right away, you can use the checkpoint we used for the results in our paper:
[checkpoint.zip](http://download.magenta.tensorflow.org/models/onsets_frames_transcription/checkpoint.zip). After unzipping
that checkpoint, you can run the following command:

```bash
CHECKPOINT_DIR=<path to unzipped checkpoint>
TEST_EXAMPLES=<path to maps_config2_test.tfrecord generated during dataset creation>
RUN_DIR=<path where output should be saved>

onsets_frames_transcription_infer \
  --acoustic_run_dir="${CHECKPOINT_DIR} \
  --examples_path="${TEST_EXAMPLES}" \
  --run_dir="${RUN_DIR}"
```

You can check on the metrics resulting from inference using TensorBoard:

```bash
tensorboard --logdir="${RUN_DIR}"
```

## Train your own

You can train your own transcription model using the training TFRecord file generated during dataset creation.

```bash
TRAIN_EXAMPLES=<path to maps_config2_train.tfrecord generated during dataset creation>
RUN_DIR=<path where checkpoints and summary events should be saved>

onsets_frames_transcription_train \
  --examples_path="${TRAIN_EXAMPLES}" \
  --run_dir="${RUN_DIR}" \
  --mode='train'
```

You can also run an eval job during training to check approximate metrics:

```bash
TEST_EXAMPLES=<path to maps_config2_test.tfrecord generated during dataset creation>
RUN_DIR=<path where checkpoints should be loaded and summary events should be saved>

onsets_frames_transcription_train \
  --examples_path="${TRAIN_EXAMPLES}" \
  --run_dir="${RUN_DIR}" \
  --mode='eval'
```

During training, you can check on progress using TensorBoard:

```bash
tensorboard --logdir="${RUN_DIR}"
```

To get final performance metrics for the model, run the `onsets_frames_transcription_infer` script as described above.
