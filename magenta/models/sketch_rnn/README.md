# Sketch-RNN: A Generative Model for Vector Drawings

Before jumping in on any code examples, please first set up your [Magenta environment](/README.md).

![Example Images](https://cdn.rawgit.com/tensorflow/magenta/master/magenta/models/sketch_rnn/assets/sketch_rnn_examples.svg)

*Examples of vector images produced by this generative model.*

This repo contains the TensorFlow code for `sketch-rnn`, the recurrent neural network model described in [Teaching Machines to Draw](https://research.googleblog.com/2017/04/teaching-machines-to-draw.html) and [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477).
We've also provided a Jupyter notebook [Sketch_RNN.ipynb](https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/Sketch_RNN.ipynb)
in our [Magenta Demos](https://github.com/tensorflow/magenta-demos) repository which demonstrates many of the examples discussed here.



# Overview of Model

`sketch-rnn` is a Sequence-to-Sequence Variational Autoencoder. The encoder RNN is a bi-directional RNN, and the decoder is an autoregressive mixture-density RNN. You can specify the type of RNN cell to use, and the size of the RNN using the settings `enc_model`, `dec_model`, `enc_size`, `dec_size`.

The encoder will sample a latent code *z*, a vector of floats with a dimension of `z_size`. Like in the VAE, we can enforce a Gaussian IID distribution to *z*, and the strength of the KL Divergence loss term is controlled using `kl_weight`. There will be a tradeoff between KL Divergence Loss and the Reconstruction Loss. We also allow some room for the latent code to store information, and not be pure Gaussian IID. Once the KL Loss term gets below `kl_tolerance`, we will stop optimizing for this term.

![Model Schematic](https://cdn.rawgit.com/tensorflow/magenta/master/magenta/models/sketch_rnn/assets/sketch_rnn_schematic.svg)

For small to medium sized datasets, dropout and data augmentation is very useful technique to avoid overfitting. We have provided options for input dropout, output dropout, and [recurrent dropout without memory loss](https://arxiv.org/abs/1603.05118). In practice, we only use recurrent dropout, and usually set it to between 65% to 90% depending on the dataset. [Layer Normalization](https://arxiv.org/abs/1607.06450) and [Recurrent Dropout](https://arxiv.org/abs/1603.05118) can be used together, forming a powerful combination for training recurrent neural nets on a small dataset.

There are two data augmentation techniques provided. The first one is a `random_scale_factor` to randomly scale the size of training images. The second augmentation technique (not used in the `sketch-rnn` paper) is dropping out random points in a line stroke. Given a line segment with more than 2 points, we can randomly drop points inside the line segments with a small probability of `augment_stroke_prob`, and still maintain a similar-looking vector image. This type of data augmentation is very powerful when used on small datasets, and is unique to vector drawings, since it is difficult to dropout random characters or notes in text or midi data, and also not possible to dropout random pixels without causing large visual differences in pixel image data. We usually set both data augmentation parameters to 10% to 20%. If there is virtually no difference for a human audience when they compare an augmented example compared to a normal example, we apply both data augmentation techniques regardless of the size of the training dataset.

Using dropout and data augmentation effectively will avoid overfitting to a small training set.

# Training a Model

To train the model you first need a dataset containing train/validation/test examples. We have provided links to the `aaron_sheep` dataset and the model will use this lightweight dataset by default.

Example Usage:
--------------

```bash
sketch_rnn_train --log_root=checkpoint_path --data_dir=dataset_path --hparams="data_set=[dataset_filename.npz]"
```

We recommend you create subdirectories inside `models` and `datasets` to hold your own data and checkpoints. The [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) logs will be stored inside `checkpoint_path` for viewing training curves for the various losses on train/validation/test datasets.

Here is a list of full options for the model, along with the default settings:

```python
data_set=['aaron_sheep.npz'],  # Our dataset. Can be list of multiple .npz sets.
num_steps=10000000,            # Total number of training set. Keep large.
save_every=500,                # Number of batches per checkpoint creation.
dec_rnn_size=512,              # Size of decoder.
dec_model='lstm',              # Decoder: lstm, layer_norm or hyper.
enc_rnn_size=256,              # Size of encoder.
enc_model='lstm',              # Encoder: lstm, layer_norm or hyper.
z_size=128,                    # Size of latent vector z. Recommend 32, 64 or 128.
kl_weight=0.5,                 # KL weight of loss equation. Recommend 0.5 or 1.0.
kl_weight_start=0.01,          # KL start weight when annealing.
kl_tolerance=0.2,              # Level of KL loss at which to stop optimizing for KL.
batch_size=100,                # Minibatch size. Recommend leaving at 100.
grad_clip=1.0,                 # Gradient clipping. Recommend leaving at 1.0.
num_mixture=20,                # Number of mixtures in Gaussian mixture model.
learning_rate=0.001,           # Learning rate.
decay_rate=0.9999,             # Learning rate decay per minibatch.
kl_decay_rate=0.99995,         # KL annealing decay rate per minibatch.
min_learning_rate=0.00001,     # Minimum learning rate.
use_recurrent_dropout=True,    # Recurrent Dropout without Memory Loss. Recomended.
recurrent_dropout_prob=0.90,   # Probability of recurrent dropout keep.
use_input_dropout=False,       # Input dropout. Recommend leaving False.
input_dropout_prob=0.90,       # Probability of input dropout keep.
use_output_dropout=False,      # Output droput. Recommend leaving False.
output_dropout_prob=0.90,      # Probability of output dropout keep.
random_scale_factor=0.15,      # Random scaling data augmention proportion.
augment_stroke_prob=0.10,      # Point dropping augmentation proportion.
conditional=True,              # If False, use decoder-only model.
```

Here are some options you may want to use to train the model on a very large dataset spanning three `.npz` files, and use [HyperLSTM](https://arxiv.org/abs/1609.09106) as the RNN cells.  For small datasets of less than 10K training examples, LSTM with Layer Normalization (`layer_norm` for both `enc_model` and `dec_model`) works best.

```bash
sketch_rnn_train --log_root=models/big_model --data_dir=datasets/big_dataset --hparams="data_set=[class1.npz,class2.npz,class3.npz],dec_model=hyper,dec_rnn_size=2048,enc_model=layer_norm,enc_rnn_size=512,save_every=5000,grad_clip=1.0,use_recurrent_dropout=0"
```

We have tested this model on TensorFlow 1.0 and 1.1 for Python 2.7.

# Datasets

Due to size limitations, this repo does not contain any datasets.

We have prepared many datasets that work out of the box with Sketch-RNN.  The Google [QuickDraw](https://quickdraw.withgoogle.com/data) Dataset is a collection of 50M vector sketches across 345 categories.  In the repo of [quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset), there is a section called *Sketch-RNN QuickDraw Dataset* that describes the pre-processed datafiles that can be used with this project.  Each category class is stored in its own file, such as `cat.npz`, and contains training/validation/test set sizes of 70000/2500/2500 examples.

You download the `.npz` datasets from [google cloud](https://console.cloud.google.com/storage/quickdraw_dataset/sketchrnn) for local use.  We recommend you create a sub directory called `datasets/quickdraw`, and save these `.npz` files in this sub directory.

In addition to the QuickDraw dataset, we have also tested this model on smaller datasets.  In the [sketch-rnn-datasets](https://github.com/hardmaru/sketch-rnn-datasets) repo, there are 3 other datasets: Aaron Koblin Sheep Market, Kanji, and Omniglot.  We recommend you create a sub directory for each of these dataset, such as `datasets/aaron_sheep`, if you wish to use them locally.  As mentioned before, recurrent dropout and data augmentation should be used when training models on small datasets to avoid overfitting.

# Creating Your Own Dataset

Please create your own interesting datasets and train this algorithm on them!  Getting your hands dirty and creating new datasets is part of the fun.  Why settle on existing pre-packaged datasets when you are potentially sitting on an interesting dataset of vector line drawings?  In our experiments, a dataset size consisting of a few thousand examples was sufficient to produce some meaningful results.  Here, we describe the format of the dataset files the model expects to see.

Each example in the dataset is stored as list of coordinate offsets: ∆x, ∆y, and a binary value representing whether the pen is lifted away from the paper.  This format, we refer to as *stroke-3*, is described in this [paper](https://arxiv.org/abs/1308.0850).  Note that the data format described in the paper has 5 elements (*stroke-5* format), and this conversion is done automatically inside the `DataLoader`.  Below is an example sketch of a turtle using this format:

![Example Training Sketches](https://cdn.rawgit.com/tensorflow/magenta/master/magenta/models/sketch_rnn/assets/data_format.svg)
*Figure: A sample sketch, as a sequence of (∆x, ∆y, binary pen state) points and in rendered form.
In the rendered sketch, the line color corresponds to the sequential stroke ordering.*

In our datasets, each example in the list of examples is represented as a `np.array` with `np.int16` datatypes.  You can store them as `np.int8` if you can get away with it to save storage space.  If your data must be in floating-point format, `np.float16` also works.  `np.float32` can be a waste of storage space.  In our data, the ∆x and ∆y offsets are represented in pixel locations, which are larger than the range of numbers a neural network model likes to see, so there is a normalization scaling process built into the model.  When we load the training data, the model will automatically convert to `np.float` and normalize accordingly before training.

If you want to create your own dataset, you must create three lists of examples for training/validation/test sets, to avoid overfitting to the training set.  The model will handle the early stopping using the validation set.  For the `aaron_sheep` dataset, we used a split of 7400/300/300 examples, and put each inside python lists called `train_data`, `validation_data`, and `test_data`.  Afterwards, we created a subdirectory called `datasets/aaron_sheep` and we use the built-in `savez_compressed` method to save a compressed version of the dataset in a `aaron_sheep.npz` file.  In all of our experiments, the size of each dataset is an exact multiple of 100, and use a `batch_size` of 100.  Deviate at your own peril.

```python
filename = os.path.join('datasets/your_dataset_directory', 'your_dataset_name.npz')
np.savez_compressed(filename, train=train_data, valid=validation_data, test=test_data)
```

We also performed simple stroke simplification to preprocess the data, called [Ramer-Douglas-Peucker](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).  There is some easy-to-use open source code for applying this algorithm [here](https://github.com/fhirschmann/rdp).  In practice, we can set the `epsilon` parameter to a value between 0.2 to 3.0, depending on how aggressively we want to simply the lines.  In the paper we used an `epsilon` parameter of 2.0. We suggest you build a dataset where the maximum sequence length is less than 250.

If you have a large set of simple SVG images, there are some available [libraries](https://pypi.python.org/pypi/svg.path) to convert subsets of SVGs into line segments, and you can then apply RDP on the line segments before converting the data to *stroke-3* format.

# Pre-Trained Models

We have provided pre-trained models for the `aaron_sheep` dataset, for both conditional and unconditional training mode, using vanilla LSTM cells and LSTM cells with [Layer Normalization](https://arxiv.org/abs/1607.06450).  These models will be downloaded by running the Jupyter Notebook.  They are stored in:

* `/tmp/sketch_rnn/models/aaron_sheep/lstm`
* `/tmp/sketch_rnn/models/aaron_sheep/lstm_uncond`
* `/tmp/sketch_rnn/models/aaron_sheep/layer_norm`
* `/tmp/sketch_rnn/models/aaron_sheep/layer_norm_uncond`

In addition, we have provided pre-trained models for selected QuickDraw datasets:

* `/tmp/sketch_rnn/models/owl/lstm`
* `/tmp/sketch_rnn/models/flamingo/lstm_uncond`
* `/tmp/sketch_rnn/models/catbus/lstm`
* `/tmp/sketch_rnn/models/elephantpig/lstm`

# Using a Model with Jupyter Notebook

![Example Images](https://cdn.rawgit.com/tensorflow/magenta/master/magenta/models/sketch_rnn/assets/catbus.svg)

*Let's get the model to interpolate between a cat and a bus!*

We've included a simple [Jupyter Notebook](https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/Sketch_RNN.ipynb) to show you how to load a pre-trained model and generate vector sketches.  You will be able to encode, decode, and morph between two vector images, and also generate new random ones.  When sampling images, you can tune the `temperature` parameter to control the level of uncertainty.

# Citation

If you find this project useful for academic purposes, please cite it as:

```
@ARTICLE{sketchrnn,
  author          = {{Ha}, David and {Eck}, Douglas},
  title           = "{A Neural Representation of Sketch Drawings}",
  journal         = {ArXiv e-prints},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1704.03477},
  primaryClass    = "cs.NE",
  keywords        = {Computer Science - Neural and Evolutionary Computing, Computer Science - Learning, Statistics - Machine Learning},
  year            = 2017,
  month           = apr,
}
```

[arXiv]: https://arxiv.org/abs/1704.03477
[blog]: https://research.googleblog.com/2017/04/teaching-machines-to-draw.html
[dataset]: https://magenta.tensorflow.org/datasets/sketchrnn
