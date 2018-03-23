"""1-D convolutional neural network glyph classifier model.

Convolves a filter horizontally along a staffline, to classify glyphs at each
x position.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr.protobuf import musicscore_pb2

# Count every glyph type except for UNKNOWN_TYPE.
NUM_GLYPHS = len(musicscore_pb2.Glyph.Type.values()) - 1


# TODO(ringwalt): Make this extend BaseGlyphClassifier.
class NeuralNetworkGlyphClassifier(object):
  """Holds a TensorFlow NN model used for classifying glyphs on staff lines."""

  def __init__(self,
               input_placeholder,
               hidden_layer,
               reconstruction_layer=None,
               autoencoder_vars=None,
               labels_placeholder=None,
               prediction_layer=None,
               prediction_vars=None):
    """Builds the NeuralNetworkGlyphClassifier that holds the TensorFlow model.

    Args:
      input_placeholder: A tf.placeholder representing the input staffline
          image. Dtype float32 and shape (batch_size, target_height, None).
      hidden_layer: An inner layer in the model. Should be the last layer in the
          autoencoder model before reconstructing the input, and/or an
          intermediate layer in the prediction network. self is intended to be
          the last common ancestor of the reconstruction_layer output and the
          prediction_layer output, if both are present.
      reconstruction_layer: The reconstruction of the input, for an autoencoder
          model. If non-None, should have the same shape as input_placeholder.
      autoencoder_vars: The variables for the autoencoder model (parameters
          affecting hidden_layer and reconstruction_layer), or None. If
          non-None, a dict mapping variable name to tf.Variable object.
      labels_placeholder: The labels tensor. A placeholder will be created if
          None is given. Dtype int32 and shape (batch_size, width). Values are
          between 0 and NUM_GLYPHS - 1 (where each value is the Glyph.Type enum
          value minus one, to skip UNKNOWN_TYPE).
      prediction_layer: The logit probability of each glyph for each column.
          Must be able to be passed to tf.nn.softmax to produce the probability
          of each glyph. 2D (width, NUM_GLYPHS). May be None if the model is not
          being used for classification.
      prediction_vars: The variables for the classification model (parameters
          affecting hidden_layer and prediction_layer), or None. If non-None, a
          dict mapping variable name to tf.Variable object.
    """
    self.input_placeholder = input_placeholder
    self.hidden_layer = hidden_layer
    self.reconstruction_layer = reconstruction_layer
    self.autoencoder_vars = autoencoder_vars or {}
    # Calculate the loss that will be minimized for the autoencoder model.
    self.autoencoder_loss = None
    if self.reconstruction_layer is not None:
      self.autoencoder_loss = (tf.reduce_mean(
          tf.squared_difference(self.input_placeholder,
                                self.reconstruction_layer)))
    self.prediction_layer = prediction_layer
    self.prediction_vars = prediction_vars or {}
    self.labels_placeholder = (labels_placeholder
                               if labels_placeholder is not None else
                               tf.placeholder(tf.int32, (None, None)))
    # Calculate the loss that will be minimized for the prediction model.
    self.prediction_loss = None
    if self.prediction_layer is not None:
      self.prediction_loss = (tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              logits=self.prediction_layer,
              labels=tf.one_hot(self.labels_placeholder, NUM_GLYPHS))))
    # The probabilities of each glyph for each column.
    self.prediction = tf.nn.softmax(self.prediction_layer)

  def get_autoencoder_initializers(self):
    """Gets the autoencoder initializer ops.

    Returns:
      The list of TensorFlow ops which initialize the autoencoder model.
    """
    return [var.initializer for var in self.autoencoder_vars.values()]

  def get_classifier_initializers(self):
    """Gets the classifier initializer ops.

    Returns:
      The list of TensorFlow ops which initialize the classifier model.
    """
    return [var.initializer for var in self.prediction_vars.values()]

  @staticmethod
  def semi_supervised_model(batch_size,
                            target_height,
                            input_placeholder=None,
                            labels_placeholder=None):
    """Constructs the semi-supervised model.

    Consists of an autoencoder and classifier, sharing a hidden layer.

    Args:
      batch_size: The number of staffline images in a batch, which must be known
          at model definition time. int.
      target_height: The height of each scaled staffline image. int.
      input_placeholder: The input layer. A placeholder will be created if None
          is given. Dtype float32 and shape (batch_size, target_height,
          any_width).
      labels_placeholder: The labels tensor. A placeholder will be created if
          None is given. Dtype int32 and shape (batch_size, width).

    Returns:
      A NeuralNetworkGlyphClassifier instance holding the model.
    """
    if input_placeholder is None:
      input_placeholder = tf.placeholder(tf.float32,
                                         (batch_size, target_height, None))
    autoencoder_vars = {}
    prediction_vars = {}

    hidden, layer_vars = InputConvLayer(input_placeholder, 10).get()
    autoencoder_vars.update(layer_vars)
    prediction_vars.update(layer_vars)

    hidden, layer_vars = HiddenLayer(hidden, 10, 10).get()
    autoencoder_vars.update(layer_vars)
    prediction_vars.update(layer_vars)

    reconstruction, layer_vars = ReconstructionLayer(
        hidden, target_height, target_height).get()
    autoencoder_vars.update(layer_vars)

    hidden, layer_vars = HiddenLayer(hidden, 10, 10, name="hidden_2").get()
    prediction_vars.update(layer_vars)

    prediction, layer_vars = PredictionLayer(hidden).get()
    prediction_vars.update(layer_vars)

    return NeuralNetworkGlyphClassifier(
        input_placeholder,
        hidden,
        reconstruction_layer=reconstruction,
        autoencoder_vars=autoencoder_vars,
        labels_placeholder=labels_placeholder,
        prediction_layer=prediction,
        prediction_vars=prediction_vars)


class BaseLayer(object):

  def __init__(self, filter_size, n_in, n_out, name):
    self.weights = tf.Variable(
        tf.truncated_normal((filter_size, n_in, n_out)), name=name + "_W")
    self.bias = tf.Variable(tf.zeros(n_out), name=name + "_bias")
    self.vars = {self.weights.name: self.weights, self.bias.name: self.bias}

  def get(self):
    """Gets the layer output and variables.

    Returns:
      The output tensor of the layer.
      The dict of variables (parameters) for the layer.
    """
    return self.output, self.vars


class InputConvLayer(BaseLayer):
  """Convolves the input image strip, producing multiple outputs per column."""

  def __init__(self, image, n_hidden, activation=tf.nn.sigmoid, name="input"):
    """Creates the InputConvLayer.

    Args:
      image: The input image (height, width). Should be wider than it is tall.
      n_hidden: The number of output nodes of the layer.
      activation: Callable applied to the convolved image. Applied to the 1D
          convolution result to produce the activation of the layer.
      name: The prefix for variable names for the layer.

    Produces self.output with shape (width, n_hidden).
    """
    height = int(image.get_shape()[1])
    super(InputConvLayer, self).__init__(
        filter_size=height, n_in=height, n_out=n_hidden, name=name)
    self.input = image
    # Transpose the image, so that the rows are "channels" in a 1D input.
    self.output = activation(
        tf.nn.conv1d(
            tf.transpose(image, [0, 2, 1]),
            self.weights,
            stride=1,
            padding="SAME") + self.bias[None, None, :])


class HiddenLayer(BaseLayer):
  """Performs a 1D convolution between hidden layers in the model."""

  def __init__(self,
               layer_in,
               filter_size,
               n_out,
               activation=tf.nn.sigmoid,
               name="hidden"):
    """Performs a 1D convolution between hidden layers in the model.

    Args:
      layer_in: The input layer (width, num_channels).
      filter_size: The width of the convolution filter.
      n_out: The number of output channels.
      activation: Callable applied to the convolved image. Applied to the 1D
          convolution result to produce the activation of the layer.
      name: The prefix for variable names for the layer.

    Produces self.output with shape (width, n_out).
    """
    n_in = int(layer_in.get_shape()[2])
    super(HiddenLayer, self).__init__(filter_size, n_in, n_out, name)
    self.output = activation(
        tf.nn.conv1d(layer_in, self.weights, stride=1, padding="SAME") +
        self.bias[None, None, :])


class ReconstructionLayer(BaseLayer):
  """Outputs a reconstructed layer."""

  def __init__(self,
               layer_in,
               filter_size,
               out_height,
               activation=tf.nn.sigmoid,
               name="reconstruction"):
    """Outputs a reconstructed image of shape (out_height, width).

    Args:
      layer_in: The input layer (width, num_channels).
      filter_size: The width of the convolution filter.
      out_height: The height of the output image.
      activation: Callable applied to the convolved image. Applied to the 1D
          convolution result to produce the activation of the output.
      name: The prefix for variable names for the layer.

    Produces self.output with shape (width, n_out).
    """
    n_in = int(layer_in.get_shape()[2])
    super(ReconstructionLayer, self).__init__(filter_size, n_in, out_height,
                                              name)
    output = activation(
        tf.nn.conv1d(layer_in, self.weights, stride=1, padding="SAME") +
        self.bias[None, None, :])
    self.output = tf.transpose(output, [0, 2, 1])


class PredictionLayer(BaseLayer):
  """Classifies each column from a hidden layer."""

  def __init__(self, layer_in, name="prediction"):
    """Outputs logit predictions for each column from a hidden layer.

    Args:
      layer_in: The input layer (width, num_channels).
      name: The prefix for variable names for the layer.

    Produces the logits for each class in self.output. Shape (width, NUM_GLYPHS)
    """
    n_in = int(layer_in.get_shape()[2])
    n_out = NUM_GLYPHS
    super(PredictionLayer, self).__init__(1, n_in, n_out, name)

    input_shape = tf.shape(layer_in)
    input_columns = tf.reshape(
        layer_in, [input_shape[0] * input_shape[1], input_shape[2]])
    # Ignore the 0th axis of the weights (convolutional filter, which is 1 here)
    weights = self.weights[0, :, :]
    output = tf.matmul(input_columns, weights) + self.bias
    self.output = tf.reshape(output,
                             [input_shape[0], input_shape[1], NUM_GLYPHS])
