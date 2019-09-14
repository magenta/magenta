# Style Transfer

Style transfer is the task of producing a pastiche image 'p' that shares the
content of a content image 'c' and the style of a style image 's'. This code
implements the paper "A Learned Representation for Artistic Style":

[A Learned Representation for Artistic Style](https://arxiv.org/abs/1610.07629). *Vincent Dumoulin, Jon Shlens,
Manjunath Kudlur*.

# Setup
Whether you want to stylize an image with one of our pre-trained models or train your own model, you need to set up your [Magenta environment](/README.md).

# Jupyter notebook
There is a Jupyter notebook [Image_Stylization.ipynb](https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/Image_Stylization.ipynb)
in our [Magenta Demos](https://github.com/tensorflow/magenta-demos) repository showing how to apply style transfer from a trained model.

# Stylizing an Image
First, download one of our pre-trained models:

* [Monet](http://download.magenta.tensorflow.org/models/multistyle-pastiche-generator-monet.ckpt)
* [Varied](http://download.magenta.tensorflow.org/models/multistyle-pastiche-generator-varied.ckpt)

(You can also train your own model, but if you're just getting started we recommend using a pre-trained model first.)

Then, run the following command:

```bash
$ image_stylization_transform \
      --num_styles=<NUMBER_OF_STYLES> \
      --checkpoint=/path/to/model.ckpt \
      --input_image=/path/to/image.jpg \
      --which_styles="[0,1,2,5,14]" \
      --output_dir=/tmp/image_stylization/output \
      --output_basename="stylized"
```

You'll have to specify the correct number of styles for the model you're using. For the Monet model this is 10 and for the varied model this is 32. The `which_styles` argument should be a Python list of integer style indices.

`which_styles` can also be used to specify a linear combination of styles to
combine in a single image. Use a Python dictionary that maps the style index to
the weights for each style. If the style index is unspecified then it will have
a zero weight. Note that the weights are not normalized.

Here's an example that produces a stylization that is an average of all of the
monet styles.

```bash
$ image_stylization_transform \
      --num_styles=10 \
      --checkpoint=multistyle-pastiche-generator-monet.ckpt \
      --input_image=photo.jpg \
      --which_styles="{0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.1,7:0.1,8:0.1,9:0.1}" \
      --output_dir=/tmp/image_stylization/output \
      --output_basename="all_monet_styles"
```

# Training a Model
To train your own model, you'll need three things:

1. A directory of images to use as styles.
2. A [trained VGG model checkpoint](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
3. The ImageNet dataset. Instructions for downloading the dataset can be found [here](https://github.com/tensorflow/models/tree/master/research/inception#getting-started).

First, you need to prepare your style images:

```bash
$ image_stylization_create_dataset \
      --vgg_checkpoint=/path/to/vgg_16.ckpt \
      --style_files=/path/to/style/images/*.jpg \
      --output_file=/tmp/image_stylization/style_images.tfrecord
```

Then, to train a model:

```bash
$ image_stylization_train \
      --train_dir=/tmp/image_stylization/run1/train
      --style_dataset_file=/tmp/image_stylization/style_images.tfrecord \
      --num_styles=<NUMBER_OF_STYLES> \
      --vgg_checkpoint=/path/to/vgg_16.ckpt \
      --imagenet_data_dir=/path/to/imagenet-2012-tfrecord
```

To evaluate the model:

```bash
$ image_stylization_evaluate \
      --style_dataset_file=/tmp/image_stylization/style_images.tfrecord \
      --train_dir=/tmp/image_stylization/run1/train \
      --eval_dir=/tmp/image_stylization/run1/eval \
      --num_styles=<NUMBER_OF_STYLES> \
      --vgg_checkpoint=/path/to/vgg_16.ckpt \
      --imagenet_data_dir=/path/to/imagenet-2012-tfrecord \
      --style_grid
```

You can also finetune a pre-trained model for new styles:

```bash
$ image_stylization_finetune \
      --checkpoint=/path/to/model.ckpt \
      --train_dir=/tmp/image_stylization/run2/train
      --style_dataset_file=/tmp/image_stylization/style_images.tfrecord \
      --num_styles=<NUMBER_OF_STYLES> \
      --vgg_checkpoint=/path/to/vgg_16.ckpt \
      --imagenet_data_dir=/path/to/imagenet-2012-tfrecord
```

# Image Stylization on Mobile
For better performance on mobile, you should train a slimmed down version of the model by specifying an alpha multiplier at training. This is an [output sample](sample_images/benchmark.png) of the models trained with different alpha values.

Here are the steps to train and convert the model to TensorFlow Lite to run on mobile:

1. (Optional but Recommended) Train a slimmed down version of the model by specifying an alpha multiplier:

```bash
$ image_stylization_train \
      --train_dir=/tmp/image_stylization/run1/train
      --style_dataset_file=/tmp/image_stylization/style_images.tfrecord \
      --num_styles=<NUMBER_OF_STYLES> \
      --alpha=0.25 \
      --vgg_checkpoint=/path/to/vgg_16.ckpt \
      --imagenet_data_dir=/path/to/imagenet-2012-tfrecord
```

2. Convert the trained model to TensorFlow Lite:

```bash
$ image_stylization_convert_tflite \
      --checkpoint=/tmp/image_stylization/run1/train \
      --num_styles=<NUMBER_OF_STYLES> \
      --alpha=0.25 \
      --output_model='/tmp/image_stylization/model.tflite'
```
