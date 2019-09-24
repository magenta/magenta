# SVG VAE

SVG VAE is a [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
problem for generating font SVGs.

This is your main access point for the model described in
[A Learned Representation for Scalable Vector Graphics]
(https://arxiv.org/abs/1904.02632).

To run any of the below commands, you first need to install Magenta as described
[here](/README.md#development-environment)

## Sample from a pretrained model

You can download model checkpoints from
[here](https://storage.googleapis.com/magentadata/models/svg_vae/svg_vae.tar.gz)
where you'll find a tarball with the models used in the paper, as well as
trained on the externally-released data described below.

After you compile the dataset (see instructions below), you can use the given
colab ([coming soon](https://github.com/tensorflow/magenta-demos))
to sample from the model. Simply modify the variables and run all cells:
```
problem = 'glyph_azzn_problem'
data_dir = '/path/to/glyphazzn_final_t2t_dataset/'
model_name = 'svg_decoder'  # or image_vae
hparam_set = 'svg_decoder'  # or image_vae
hparams = ('vae_ckpt_dir=/path/to/saved_models/image_vae,' +
           'vae_data_dir=/path/to/glyphazzn_final_t2t_dataset/')
ckpt_dir = '/path/to/saved_models/svg_decoder'
```

## Train your own

Training your own model consists of two main steps:

1. Data generation / preprocessing
1. Training

These two steps are performed by the `t2t_datagen` and `t2t_trainer` scripts,
respectively.

### Data generation / preprocessing

In order to re-create the dataset, which we refer to here as `glyphazzn` (glyphs
A-z, 0-9), you first need to collect the fonts available in the urls listed in
[this file](https://storage.googleapis.com/magentadata/models/svg_vae/glyphazzn_urls.txt).

Store these in a parquetio database with columns:
```
{'uni': int64,  # unicode value of this glyph
 'width': int64,  # width of this glyph's viewport (provided by fontforge)
 'vwidth': int64,  # vertical width of this glyph's viewport
 'sfd': binary/str,  # glyph, converted to .sfd format, with a single SplineSet
 'id': binary/str,  # id of this glyph
 'binary_fp': binary/str}  # font identifier (provided in glyphazzn_urls.txt)
```
Your end result should be a directory with a bunch of files like
`/path/to/glyphs-parquetio-00xxx-of-0xxxx`
This database is the raw data, which our `datagen_beam.py` will process.

To process the raw data, build `datagen_beam.par`, and run it in a compute
cluster with [beam]
(https://beam.apache.org/documentation/runners/capability-matrix/), such as
[Google Cloud Dataflow]
(https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python).
```
datagen_beam.par --logtostderr \
--raw_data_file=/path/to/glyphs-parquetio
--final_data_file=/path/to/tfrecord_dataset
--final_stats_file=/path/to/tfrecord_stats
```
This will create dataset and stats files in TFRecord format.
The stats files has a mean and stdev of the vectorized paths in each glyph. They
will be used to normalize input to the model at train time.

These files are almost ready for use, but first we have to run `t2t_datageen`.
This will create a few extra fields that required by tensor2tensor for training.
To do this, add `/path/to/tfrecord_dataset/` and `/path/to/tfrecord_stats` to
the top of glyphazzn.py, like so:
```
RAW_STAT_FILE = '/path/to/tfrecord_stats-00000-of-00001'
RAW_DATA_FILES = '/path/to/tfrecord_data*'
```
Then run `t2t_datagen`:
```
t2t_datagen --logtostderr \
  --data_dir /path/to/glyphazzn_final_t2t_dataset \
  --tmp_dir /tmp/t2t_datagen \
  --problem glyph_azzn_problem
```

You're done re-creating the dataset, and ready to train the models.


### Training

To train the VAE, run:
```
t2t_trainer --logtostderr \
  --problem glyph_azzn_problem \
  --data_dir /path/to/glyphazzn_final_t2t_dataset \
  --output_dir /path/to/saved_models/image_vae \
  --model image_vae \
  --hparams_set image_vae \
  --train_steps 100000
```

After the vae is done training, train the SVG decoder like so:
```
t2t_trainer --logtostderr \
  --problem glyph_azzn_problem \
  --data_dir /path/to/glyphazzn_final_t2t_dataset/ \
  --output_dir /path/to/saved_models/svg_decoder \
  --model svg_decoder \
  --hparams_set svg_decoder \
  --train_steps 300000 \
  --hparams="vae_ckpt_dir=/path/to/saved_models/image_vae,vae_data_dir=/path/to/glyphazzn_final_t2t_dataset/"
```
Note that if you train the vae with different hparams, you must also set
`vae_hparams`. If you change the problem, you must set `vae_problem`.
