"""Tests embedding_rnn."""

import tensorflow as tf
from magenta.models.nsynth.embedding_rnn import embedding_rnn

FLAGS = tf.app.flags.FLAGS


class EmbeddingRnnTest(tf.test.TestCase):

  def setUp(self):
    self.hps_model = embedding_rnn.default_hps()
    self.hps_model.parse(FLAGS.hparam)
    self.hps_model.data_set = 'small_train_z.npy'
    self.hps_model.rnn_size = 64
    self.hps_model.num_steps = 100
    self.hps_model.overfit = 1
    self.hps_model_values = self.hps_model.values()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def testHParams(self):
    self.assertEqual(self.hps_model_values['rnn_size'], 64)

  def testCopyModel(self):
    self.eval_hps_model_values = self.hps_model.values()
    self.assertEqual(self.hps_model_values['rnn_size'],
                     self.eval_hps_model_values['rnn_size'])


if __name__ == '__main__':
  tf.test.main()
