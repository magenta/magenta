"""Tests for the neural network glyph classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr.glyphs import neural

NUM_STAFFLINES = 5
TARGET_HEIGHT = 15
IMAGE_WIDTH = 100


class NeuralNetworkGlyphClassifierTest(tf.test.TestCase):

  def testSemiSupervisedClassifier(self):
    # Ensure that the losses can be evaluated without error.
    stafflines = tf.random_uniform((NUM_STAFFLINES, TARGET_HEIGHT, IMAGE_WIDTH))
    # Use every single glyph once except for NONE (0).
    labels_single_batch = np.concatenate([
        np.arange(neural.NUM_GLYPHS),
        np.zeros(IMAGE_WIDTH - neural.NUM_GLYPHS)
    ]).astype(np.int32)
    labels = np.repeat(labels_single_batch[None, :], NUM_STAFFLINES, axis=0)
    classifier = neural.NeuralNetworkGlyphClassifier.semi_supervised_model(
        batch_size=NUM_STAFFLINES,
        target_height=TARGET_HEIGHT,
        input_placeholder=stafflines,
        labels_placeholder=tf.constant(labels))
    with self.test_session() as sess:
      # The autoencoder must run successfully with only its vars initialized.
      # The loss must always be positive.
      sess.run(classifier.get_autoencoder_initializers())
      self.assertGreater(sess.run(classifier.autoencoder_loss), 0)

      # The classifier must run successfully with its vars initialized too.
      sess.run(classifier.get_classifier_initializers())
      self.assertGreater(sess.run(classifier.prediction_loss), 0)


if __name__ == '__main__':
  tf.test.main()
