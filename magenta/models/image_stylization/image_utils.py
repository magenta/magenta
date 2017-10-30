# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image-related functions for style transfer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tempfile


import numpy as np
import scipy
import scipy.misc
import tensorflow as tf

from magenta.models.image_stylization import imagenet_data
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops


slim = tf.contrib.slim


_EVALUATION_IMAGES_GLOB = 'evaluation_images/*.jpg'


def imagenet_inputs(batch_size, image_size, num_readers=1,
                    num_preprocess_threads=4):
  """Loads a batch of imagenet inputs.

  Used as a replacement for inception.image_processing.inputs in
  tensorflow/models in order to get around the use of hard-coded flags in the
  image_processing module.

  Args:
    batch_size: int, batch size.
    image_size: int. The images will be resized bilinearly to shape
        [image_size, image_size].
    num_readers: int, number of preprocessing threads per tower.  Must be a
        multiple of 4.
    num_preprocess_threads: int, number of parallel readers.

  Returns:
    4-D tensor of images of shape [batch_size, image_size, image_size, 3], with
    values in [0, 1].

  Raises:
    IOError: If ImageNet data files cannot be found.
    ValueError: If `num_preprocess_threads is not a multiple of 4 or
        `num_readers` is less than 1.
  """
  imagenet = imagenet_data.ImagenetData('train')

  with tf.name_scope('batch_processing'):
    data_files = imagenet.data_files()
    if data_files is None:
      raise IOError('No ImageNet data files found')

    # Create filename_queue.
    filename_queue = tf.train.string_input_producer(data_files,
                                                    shuffle=True,
                                                    capacity=16)

    if num_preprocess_threads % 4:
      raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    input_queue_memory_factor = 16
    min_queue_examples = examples_per_shard * input_queue_memory_factor
    examples_queue = tf.RandomShuffleQueue(
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    enqueue_ops = []
    for _ in range(num_readers):
      reader = imagenet.reader()
      _, value = reader.read(filename_queue)
      enqueue_ops.append(examples_queue.enqueue([value]))

    tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
    example_serialized = examples_queue.dequeue()

    images_and_labels = []
    for _ in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      image_buffer, label_index, _, _ = _parse_example_proto(
          example_serialized)
      image = tf.image.decode_jpeg(image_buffer, channels=3)

      # pylint: disable=protected-access
      image = _aspect_preserving_resize(image, image_size + 2)
      image = _central_crop([image], image_size, image_size)[0]
      # pylint: enable=protected-access
      image.set_shape([image_size, image_size, 3])
      image = tf.to_float(image) / 255.0

      images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size)

    images = tf.reshape(images, shape=[batch_size, image_size, image_size, 3])

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_index_batch, [batch_size])


def style_image_inputs(style_dataset_file, batch_size=None, image_size=None,
                       square_crop=False, shuffle=True):
  """Loads a batch of random style image given the path of tfrecord dataset.

  Args:
    style_dataset_file: str, path to the tfrecord dataset of style files.
        The dataset is produced via the create_style_dataset.py script and is
        made of Example protobufs with the following features:
        * 'image_raw': byte encoding of the JPEG string of the style image.
        * 'label': integer identifier of the style image in [0, N - 1], where
              N is the number of examples in the dataset.
        * 'vgg_16/<LAYER_NAME>': Gram matrix at layer <LAYER_NAME> of the VGG-16
              network (<LAYER_NAME> in {conv,pool}{1,2,3,4,5}) for the style
              image.
    batch_size: int. If provided, batches style images. Defaults to None.
    image_size: int. The images will be resized bilinearly so that the smallest
        side has size image_size. Defaults to None.
    square_crop: bool. If True, square-crops to [image_size, image_size].
        Defaults to False.
    shuffle: bool, whether to shuffle style files at random. Defaults to True.

  Returns:
    If batch_size is defined, a 4-D tensor of shape [batch_size, ?, ?, 3] with
    values in [0, 1] for the style image, and 1-D tensor for the style label.

  Raises:
    ValueError: if center cropping is requested but no image size is provided,
        or if batch size is specified but center-cropping is not requested.
  """
  vgg_layers = ['vgg_16/conv1', 'vgg_16/pool1', 'vgg_16/conv2', 'vgg_16/pool2',
                'vgg_16/conv3', 'vgg_16/pool3', 'vgg_16/conv4', 'vgg_16/pool4',
                'vgg_16/conv5', 'vgg_16/pool5']

  if square_crop and image_size is None:
    raise ValueError('center-cropping requires specifying the image size.')
  if batch_size is not None and not square_crop:
    raise ValueError('batching requires center-cropping.')

  with tf.name_scope('style_image_processing'):
    filename_queue = tf.train.string_input_producer(
        [style_dataset_file], shuffle=False, capacity=1,
        name='filename_queue')
    if shuffle:
      examples_queue = tf.RandomShuffleQueue(
          capacity=64,
          min_after_dequeue=32,
          dtypes=[tf.string], name='random_examples_queue')
    else:
      examples_queue = tf.FIFOQueue(
          capacity=64,
          dtypes=[tf.string], name='fifo_examples_queue')
    reader = tf.TFRecordReader()
    _, value = reader.read(filename_queue)
    enqueue_ops = [examples_queue.enqueue([value])]
    tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
    example_serialized = examples_queue.dequeue()
    features = tf.parse_single_example(
        example_serialized,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string),
                  'vgg_16/conv1': tf.FixedLenFeature([64, 64], tf.float32),
                  'vgg_16/pool1': tf.FixedLenFeature([64, 64], tf.float32),
                  'vgg_16/conv2': tf.FixedLenFeature([128, 128], tf.float32),
                  'vgg_16/pool2': tf.FixedLenFeature([128, 128], tf.float32),
                  'vgg_16/conv3': tf.FixedLenFeature([256, 256], tf.float32),
                  'vgg_16/pool3': tf.FixedLenFeature([256, 256], tf.float32),
                  'vgg_16/conv4': tf.FixedLenFeature([512, 512], tf.float32),
                  'vgg_16/pool4': tf.FixedLenFeature([512, 512], tf.float32),
                  'vgg_16/conv5': tf.FixedLenFeature([512, 512], tf.float32),
                  'vgg_16/pool5': tf.FixedLenFeature([512, 512], tf.float32)})
    image = tf.image.decode_jpeg(features['image_raw'])
    label = features['label']
    gram_matrices = [features[vgg_layer] for vgg_layer in vgg_layers]
    image.set_shape([None, None, 3])

    if image_size:
      if square_crop:
        image = _aspect_preserving_resize(image, image_size + 2)
        image = _central_crop([image], image_size, image_size)[0]
        image.set_shape([image_size, image_size, 3])
      else:
        image = _aspect_preserving_resize(image, image_size)

    image = tf.to_float(image) / 255.0

    if batch_size is None:
      image = tf.expand_dims(image, 0)
    else:
      image_label_gram_matrices = tf.train.batch(
          [image, label] + gram_matrices, batch_size=batch_size)
      image, label = image_label_gram_matrices[:2]
      gram_matrices = image_label_gram_matrices[2:]

    gram_matrices = dict([(vgg_layer, gram_matrix)
                          for vgg_layer, gram_matrix
                          in zip(vgg_layers, gram_matrices)])
    return image, label, gram_matrices


def arbitrary_style_image_inputs(style_dataset_file,
                                 batch_size=None,
                                 image_size=None,
                                 center_crop=True,
                                 shuffle=True,
                                 augment_style_images=False,
                                 random_style_image_size=False,
                                 min_rand_image_size=128,
                                 max_rand_image_size=300):
  """Loads a batch of random style image given the path of tfrecord dataset.

  This method does not return pre-compute Gram matrices for the images like
  style_image_inputs. But it can provide data augmentation. If
  augment_style_images is equal to True, then style images will randomly
  modified (eg. changes in brightness, hue or saturation) for data
  augmentation. If random_style_image_size is set to True then all images
  in one batch will be resized to a random size.
  Args:
    style_dataset_file: str, path to the tfrecord dataset of style files.
    batch_size: int. If provided, batches style images. Defaults to None.
    image_size: int. The images will be resized bilinearly so that the smallest
        side has size image_size. Defaults to None.
    center_crop: bool. If True, center-crops to [image_size, image_size].
        Defaults to False.
    shuffle: bool, whether to shuffle style files at random. Defaults to False.
    augment_style_images: bool. Wheather to augment style images or not.
    random_style_image_size: bool. If this value is True, then all the style
        images in one batch will be resized to a random size between
        min_rand_image_size and max_rand_image_size.
    min_rand_image_size: int. If random_style_image_size is True, this value
        specifies the minimum image size.
    max_rand_image_size: int. If random_style_image_size is True, this value
        specifies the maximum image size.

  Returns:
    4-D tensor of shape [1, ?, ?, 3] with values in [0, 1] for the style
    image (with random changes for data augmentation if
    augment_style_image_size is set to true), and 0-D tensor for the style
    label, 4-D tensor of shape [1, ?, ?, 3] with values in [0, 1] for the style
    image without random changes for data augmentation.

  Raises:
    ValueError: if center cropping is requested but no image size is provided,
        or if batch size is specified but center-cropping or
        augment-style-images is not requested,
        or if both augment-style-images and center-cropping are requested.
  """
  if center_crop and image_size is None:
    raise ValueError('center-cropping requires specifying the image size.')
  if center_crop and augment_style_images:
    raise ValueError(
        'When augment_style_images is true images will be randomly cropped.')
  if batch_size is not None and not center_crop and not augment_style_images:
    raise ValueError(
        'batching requires same image sizes (Set center-cropping or '
        'augment_style_images to true)')

  with tf.name_scope('style_image_processing'):
    # Force all input processing onto CPU in order to reserve the GPU for the
    # forward inference and back-propagation.
    with tf.device('/cpu:0'):
      filename_queue = tf.train.string_input_producer(
          [style_dataset_file],
          shuffle=False,
          capacity=1,
          name='filename_queue')
      if shuffle:
        examples_queue = tf.RandomShuffleQueue(
            capacity=64,
            min_after_dequeue=32,
            dtypes=[tf.string],
            name='random_examples_queue')
      else:
        examples_queue = tf.FIFOQueue(
            capacity=64, dtypes=[tf.string], name='fifo_examples_queue')
      reader = tf.TFRecordReader()
      _, value = reader.read(filename_queue)
      enqueue_ops = [examples_queue.enqueue([value])]
      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
      features = tf.parse_single_example(
          example_serialized,
          features={
              'label': tf.FixedLenFeature([], tf.int64),
              'image_raw': tf.FixedLenFeature([], tf.string)
          })
      image = tf.image.decode_jpeg(features['image_raw'])
      image.set_shape([None, None, 3])
      label = features['label']

      if image_size is not None:
        image_channels = image.shape[2].value
        if augment_style_images:
          image_orig = image
          image = tf.image.random_brightness(image, max_delta=0.8)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_flip_left_right(image)
          image = tf.image.random_flip_up_down(image)
          random_larger_image_size = random_ops.random_uniform(
              [],
              minval=image_size + 2,
              maxval=image_size + 200,
              dtype=dtypes.int32)
          image = _aspect_preserving_resize(image, random_larger_image_size)
          image = tf.random_crop(
              image, size=[image_size, image_size, image_channels])
          image.set_shape([image_size, image_size, image_channels])

          image_orig = _aspect_preserving_resize(image_orig, image_size + 2)
          image_orig = _central_crop([image_orig], image_size, image_size)[0]
          image_orig.set_shape([image_size, image_size, 3])
        elif center_crop:
          image = _aspect_preserving_resize(image, image_size + 2)
          image = _central_crop([image], image_size, image_size)[0]
          image.set_shape([image_size, image_size, image_channels])
          image_orig = image
        else:
          image = _aspect_preserving_resize(image, image_size)
          image_orig = image

      image = tf.to_float(image) / 255.0
      image_orig = tf.to_float(image_orig) / 255.0

      if batch_size is None:
        image = tf.expand_dims(image, 0)
      else:
        [image, image_orig, label] = tf.train.batch(
            [image, image_orig, label], batch_size=batch_size)

      if random_style_image_size:
        # Selects a random size for the style images and resizes all the images
        # in the batch to that size.
        image = _aspect_preserving_resize(image,
                                          random_ops.random_uniform(
                                              [],
                                              minval=min_rand_image_size,
                                              maxval=max_rand_image_size,
                                              dtype=dtypes.int32))

      return image, label, image_orig


def load_np_image(image_file):
  """Loads an image as a numpy array.

  Args:
    image_file: str. Image file.

  Returns:
    A 3-D numpy array of shape [image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  return np.float32(load_np_image_uint8(image_file) / 255.0)


def load_np_image_uint8(image_file):
  """Loads an image as a numpy array.

  Args:
    image_file: str. Image file.

  Returns:
    A 3-D numpy array of shape [image_size, image_size, 3] and dtype uint8,
    with values in [0, 255].
  """
  with tempfile.NamedTemporaryFile() as f:
    f.write(tf.gfile.GFile(image_file, 'rb').read())
    f.flush()
    image = scipy.misc.imread(f.name)
    # Workaround for black-and-white images
    if image.ndim == 2:
      image = np.tile(image[:, :, None], (1, 1, 3))
    return image


def save_np_image(image, output_file, save_format='jpeg'):
  """Saves an image to disk.

  Args:
    image: 3-D numpy array of shape [image_size, image_size, 3] and dtype
        float32, with values in [0, 1].
    output_file: str, output file.
    save_format: format for saving image (eg. jpeg).
  """
  image = np.uint8(image * 255.0)
  buf = io.BytesIO()
  scipy.misc.imsave(buf, np.squeeze(image, 0), format=save_format)
  buf.seek(0)
  f = tf.gfile.GFile(output_file, 'w')
  f.write(buf.getvalue())
  f.close()


def load_image(image_file, image_size=None):
  """Loads an image and center-crops it to a specific size.

  Args:
    image_file: str. Image file.
    image_size: int, optional. Desired size. If provided, crops the image to
        a square and resizes it to the requested size. Defaults to None.

  Returns:
    A 4-D tensor of shape [1, image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  image = tf.constant(np.uint8(load_np_image(image_file) * 255.0))
  if image_size is not None:
    # Center-crop into a square and resize to image_size
    small_side = min(image.get_shape()[0].value, image.get_shape()[1].value)
    image = tf.image.resize_image_with_crop_or_pad(
        image, small_side, small_side)
    image = tf.image.resize_images(image, [image_size, image_size])
  image = tf.to_float(image) / 255.0

  return tf.expand_dims(image, 0)


def load_evaluation_images(image_size):
  """Loads images for evaluation.

  Args:
    image_size: int. Image size.

  Returns:
    Tensor. A batch of evaluation images.

  Raises:
    IOError: If no evaluation images can be found.
  """
  glob = os.path.join(tf.resource_loader.get_data_files_path(),
                      _EVALUATION_IMAGES_GLOB)
  evaluation_images = tf.gfile.Glob(glob)
  if not evaluation_images:
    raise IOError('No evaluation images found')
  return tf.concat(
      [load_image(path, image_size) for path in evaluation_images], 0)


def form_image_grid(input_tensor, grid_shape, image_shape, num_channels):
  """Arrange a minibatch of images into a grid to form a single image.

  Args:
    input_tensor: Tensor. Minibatch of images to format, either 4D
        ([batch size, height, width, num_channels]) or flattened
        ([batch size, height * width * num_channels]).
    grid_shape: Sequence of int. The shape of the image grid,
        formatted as [grid_height, grid_width].
    image_shape: Sequence of int. The shape of a single image,
        formatted as [image_height, image_width].
    num_channels: int. The number of channels in an image.

  Returns:
    Tensor representing a single image in which the input images have been
    arranged into a grid.

  Raises:
    ValueError: The grid shape and minibatch size don't match, or the image
        shape and number of channels are incompatible with the input tensor.
  """
  if grid_shape[0] * grid_shape[1] != int(input_tensor.get_shape()[0]):
    raise ValueError('Grid shape incompatible with minibatch size.')
  if len(input_tensor.get_shape()) == 2:
    num_features = image_shape[0] * image_shape[1] * num_channels
    if int(input_tensor.get_shape()[1]) != num_features:
      raise ValueError('Image shape and number of channels incompatible with '
                       'input tensor.')
  elif len(input_tensor.get_shape()) == 4:
    if (int(input_tensor.get_shape()[1]) != image_shape[0] or
        int(input_tensor.get_shape()[2]) != image_shape[1] or
        int(input_tensor.get_shape()[3]) != num_channels):
      raise ValueError('Image shape and number of channels incompatible with '
                       'input tensor.')
  else:
    raise ValueError('Unrecognized input tensor format.')
  height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
  input_tensor = tf.reshape(
      input_tensor, grid_shape + image_shape + [num_channels])
  input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
  input_tensor = tf.reshape(
      input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
  input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
  input_tensor = tf.reshape(
      input_tensor, [1, height, width, num_channels])
  return input_tensor


# The following functions are copied over from
# tf.slim.preprocessing.vgg_preprocessing
# because they're not visible to this module.
def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.strided_slice instead of crop_to_bounding box as it accepts tensors
  # to define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.strided_slice(image, offsets, offsets + cropped_shape,
                             strides=tf.ones_like(offsets))
  return tf.reshape(image, cropped_shape)


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image or a 4-D batch of images `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D or 4-D tensor containing the resized image(s).
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  input_rank = len(image.get_shape())
  if input_rank == 3:
    image = tf.expand_dims(image, 0)

  shape = tf.shape(image)
  height = shape[1]
  width = shape[2]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  if input_rank == 3:
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
  else:
    resized_image.set_shape([None, None, None, 3])
  return resized_image


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']


def center_crop_resize_image(image, image_size):
  """Center-crop into a square and resize to image_size.

  Args:
    image: A 3-D image `Tensor`.
    image_size: int, Desired size. Crops the image to a square and resizes it
      to the requested size.

  Returns:
    A 4-D tensor of shape [1, image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  shape = tf.shape(image)
  small_side = tf.minimum(shape[0], shape[1])
  image = tf.image.resize_image_with_crop_or_pad(image, small_side, small_side)
  image = tf.to_float(image) / 255.0

  image = tf.image.resize_images(image, tf.constant([image_size, image_size]))

  return tf.expand_dims(image, 0)


def resize_image(image, image_size):
  """Resize input image preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    image_size: int, desired size of the smallest size of image after resize.

  Returns:
    A 4-D tensor of shape [1, image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  image = _aspect_preserving_resize(image, image_size)
  image = tf.to_float(image) / 255.0

  return tf.expand_dims(image, 0)

