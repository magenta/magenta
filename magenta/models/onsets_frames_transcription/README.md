## Onsets and Frames: Dual-Objective Piano Transcription

State of the art piano transcription, including velocity estimation.

For model details, see our paper on arXiv:
[Onsets and Frames: Dual-Objective Piano Transcription](https://goo.gl/magenta/onsets-frames-paper). You can also listen to the [Audio Examples](https://goo.gl/magenta/onsets-frames-examples) described in the paper.

## Colab Notebook

The easiest way to use the model is with our [Onsets and Frames Colab Notebook](https://goo.gl/magenta/onsets-frames-colab). You can upload arbitrary audio files and receive a transcription without installing any software.

## Transcription Script

If you would like to run transcription locally, you can use the transcribe
script. First, set up your [Magenta environment](/README.md).

Next, download our pre-trained
[checkpoint](https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip),
which is trained on the [MAESTRO dataset](g.co/magenta/maestro).

After unzipping that checkpoint, you can run the following command:

```bash
CHECKPOINT_DIR=<path to unzipped checkpoint, should have 'train' subdir>
onsets_frames_transcription_transcribe \
  --acoustic_run_dir="${CHECKPOINT_DIR}" \
  <piano_recording1.wav, piano_recording2.wav, ...>
```

## Train your own

If you would like to train the model yourself, first set up your [Magenta environment](/README.md).

### Dataset creation

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

### Training

Now can train your own transcription model using the training TFRecord file generated during dataset creation.

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
  --examples_path="${TEST_EXAMPLES}" \
  --run_dir="${RUN_DIR}" \
  --mode='eval'
```

During training, you can check on progress using TensorBoard:

```bash
tensorboard --logdir="${RUN_DIR}"
```

### Inference

To get final performance metrics for the model, run the `onsets_frames_transcription_infer` script.

```bash
CHECKPOINT_DIR=${RUN_DIR}/train
TEST_EXAMPLES=<path to maps_config2_test.tfrecord generated during dataset creation>
RUN_DIR=<path where output should be saved>

onsets_frames_transcription_infer \
  --acoustic_run_dir="${CHECKPOINT_DIR}" \
  --examples_path="${TEST_EXAMPLES}" \
  --run_dir="${RUN_DIR}"
```

You can check on the metrics resulting from inference using TensorBoard:

```bash
tensorboard --logdir="${RUN_DIR}"
```

Note that the stats you get may differ slightly from our paper due to small
differences between our internal and external codebase.
