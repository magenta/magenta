from magenta.pipelines import pipeline
import tensorflow as tf


class EncoderPipeline(pipeline.Pipeline):
  """A pipeline that converts an EventSequence to a model encoding."""

  def __init__(self, input_type, encoder_decoder, name=None):
    """Constructs an EncoderPipeline.

    Args:
      input_type: The type this pipeline expects as input.
      encoder_decoder: An EventSequenceEncoderDecoder.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=input_type,
        output_type=tf.train.SequenceExample,
        name=name)
    self._encoder_decoder = encoder_decoder

  def transform(self, seq):
    encoded = self._encoder_decoder.encode(seq)
    return [encoded]
