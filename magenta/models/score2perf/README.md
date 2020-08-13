# Score2Perf and Music Transformer

Score2Perf is a collection of [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) problems for
generating musical performances, either unconditioned or conditioned on a
musical score.

This is your main access point for the Music Transformer model described in
[this paper](https://arxiv.org/abs/1809.04281). If you only want to experiment
with our pretrained models, check out our [Piano Transformer colab notebook](https://colab.research.google.com/notebooks/magenta/piano_transformer/piano_transformer.ipynb).

To run any of the below commands, you first need to install Magenta as described
[here](/README.md#development-environment), except you'll also need Apache Beam
so the install command is:

```
pip install -e .[beam]
```

## Train your own model

Training your own model consists of two main steps:

1. Data generation / preprocessing
1. Training

These two steps are performed by the `t2t_datagen` and `t2t_trainer` scripts,
respectively.

### Data generation / preprocessing

Data generation downloads and preprocesses the source dataset into a format
that can be easily digested by the various Tensor2Tensor models.

This is going to be a little bit annoying, but to create a dataset for training
music transformer, you'll probably want to use Cloud Dataflow or some other
platform that supports Apache Beam. You can also run datagen locally, but it
will be very slow due to the NoteSequence preprocessing.

Anyway, to prepare the dataset, do the following:

1. Set up Google Cloud Dataflow. The quickest way to do this is described in [this guide](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python).
1. Run the following command:

```
PROBLEM=score2perf_maestro_language_uncropped_aug
BUCKET=bucket_name
PROJECT=project_name

PIPELINE_OPTIONS=\
"--runner=DataflowRunner,"\
"--project=${PROJECT},"\
"--temp_location=gs://${BUCKET}/tmp,"\
"--setup_file=/path/to/setup.py"

t2t_datagen \
  --data_dir=gs://${BUCKET}/datagen \
  --problem=${PROBLEM} \
  --pipeline_options="${PIPELINE_OPTIONS}" \
  --alsologtostderr
```

This should take ~20 minutes to run and cost you maybe $0.25 in compute. After
it completes, you should see a bunch of files like `score2perf_maestro_language_uncropped_aug-{train|dev|test}.tfrecord-?????-of-?????` in the `data_dir` in the bucket you specified. Download these files to
your machine; all together they should be a little over 1 GB.

You could also train using Google Cloud. As this will be a little more expensive
and we have not tried it, the rest of this guide assumes you have downloaded the
generated TFRecord files to your local machine.


### Training

After you've downloaded the generated TFRecord files, run the following command
to train:

```
DATA_DIR=/generated/tfrecords/dir
HPARAMS_SET=score2perf_transformer_base
MODEL=transformer
PROBLEM=score2perf_maestro_language_uncropped_aug
TRAIN_DIR=/training/dir

HPARAMS=\
"label_smoothing=0.0,"\
"max_length=0,"\
"max_target_seq_length=2048"

t2t_trainer \
  --data_dir="${DATA_DIR}" \
  --hparams=${HPARAMS} \
  --hparams_set=${HPARAMS_SET} \
  --model=${MODEL} \
  --output_dir=${TRAIN_DIR} \
  --problem=${PROBLEM} \
  --train_steps=1000000
```


### Sampling from the model

Then you can use the interactive T2T decoder script to sample from the model:

```
DATA_DIR=/generated/tfrecords/dir
HPARAMS_SET=score2perf_transformer_base
MODEL=transformer
PROBLEM=score2perf_maestro_language_uncropped_aug
TRAIN_DIR=/training/dir

DECODE_HPARAMS=\
"alpha=0,"\
"beam_size=1,"\
"extra_length=2048"

t2t_decoder \
  --data_dir="${DATA_DIR}" \
  --decode_hparams="${DECODE_HPARAMS}" \
  --decode_interactive \
  --hparams="sampling_method=random" \
  --hparams_set=${HPARAMS_SET} \
  --model=${MODEL} \
  --problem=${PROBLEM} \
  --output_dir=${TRAIN_DIR}
```

Generated MIDI files will end up in your /tmp directory.
