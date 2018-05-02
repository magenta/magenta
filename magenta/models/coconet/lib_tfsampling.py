"""Defines the graph for sampling from Coconet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
# internal imports
import numpy as np
import tensorflow as tf
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_hparams
from magenta.models.coconet import lib_sampling

FLAGS = tf.app.flags.FLAGS


class CoconetSampleGraph(object):
  """Graph for Gibbs sampling from Coconet."""

  def __init__(self, chkpt_path, placeholders=None):
    self.chkpt_path = chkpt_path
    self.hparams = lib_hparams.load_hparams(chkpt_path)
    if placeholders is None:
      self.placeholders = self.get_placeholders()
    else:
      self.placeholders = placeholders

    self.build_sample_graph()
    self.sess = self.instantiate_sess_and_restore_checkpoint()

  def get_placeholders(self):
    hparams = self.hparams
    return dict(
        pianorolls=tf.placeholder(
            tf.float32,
            [None, None, hparams.num_pitches, hparams.num_instruments]),
        outer_masks=tf.placeholder(
            tf.float32,
            [None, None, hparams.num_pitches, hparams.num_instruments]),
        sample_steps=tf.placeholder(tf.float32, None),
        total_gibbs_steps=tf.placeholder(tf.float32, None),
        current_step=tf.placeholder(tf.float32, None),
        temperature=tf.placeholder(tf.float32, None))

  @property
  def inputs(self):
    return self.placeholders

  @property
  def outer_masks(self):
    return self.placeholders["outer_masks"]

  @property
  def temperature(self):
    return self.inputs["temperature"]

  def build_sample_graph(self):
    """Builds the tf.while_loop based sample graph."""
    tt = tf.shape(self.inputs["pianorolls"])[1]
    sample_steps = tf.to_float(self.inputs["sample_steps"])
    total_gibbs_steps = self.inputs["total_gibbs_steps"]

    total_gibbs_steps = tf.cond(
        tf.equal(total_gibbs_steps, 0),
        lambda: tf.to_float(tt * self.hparams.num_instruments),
        lambda: tf.to_float(total_gibbs_steps))

    sample_steps = tf.cond(
        tf.equal(sample_steps, 0),
        lambda: total_gibbs_steps,
        lambda: tf.to_float(sample_steps))
    sample_steps = tf.Print(sample_steps, [total_gibbs_steps, sample_steps],
                            "total_gibbs_steps, sample_steps")

    def infer_step(pianorolls, step_count):
      """Called by tf.while_loop, takes a Gibbs step."""
      mask_prob = compute_mask_prob_from_yao_schedule(step_count,
                                                      total_gibbs_steps)
      # 1 indicates mask out, 0 is not mask.
      masks = make_bernoulli_masks(tf.shape(pianorolls), mask_prob,
                                   self.outer_masks)

      logits = self.predict(pianorolls, masks)
      samples = sample_with_temperature(logits, temperature=self.temperature)

      outputs = pianorolls * (1 - masks) + samples * masks

      check_completion_op = tf.assert_equal(
          tf.where(tf.equal(tf.reduce_max(masks, axis=2), 1.),
                   tf.reduce_max(outputs, axis=2),
                   tf.reduce_max(pianorolls, axis=2)),
          1.)
      with tf.control_dependencies([check_completion_op]):
        outputs = tf.identity(outputs)

      step_count += 1
      return outputs, step_count

    current_step = tf.to_float(self.inputs["current_step"])

    # Initializes pianorolls by evaluating the model once to fill in all gaps.
    logits = self.predict(self.inputs["pianorolls"], self.outer_masks)
    samples = sample_with_temperature(logits, temperature=self.temperature)
    tf.get_variable_scope().reuse_variables()

    self.samples, current_step = tf.while_loop(
        lambda samples, current_step: current_step < sample_steps,
        infer_step, [samples, current_step],
        shape_invariants=[
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape(None),
        ],
        back_prop=False,
        parallel_iterations=1)

  def predict(self, pianorolls, masks):
    """Evalutes the model once and returns predictions."""
    direct_inputs = dict(
        pianorolls=pianorolls, masks=masks,
        lengths=tf.to_float([tf.shape(pianorolls)[1]]))

    model = lib_graph.build_graph(
        is_training=False,
        hparams=self.hparams,
        direct_inputs=direct_inputs,
        use_placeholders=False)
    self.logits = model.logits
    return self.logits

  def instantiate_sess_and_restore_checkpoint(self):
    sess = tf.Session()
    saver = tf.train.Saver()
    tf.logging.info("loading checkpoint %s", self.chkpt_path)
    chkpt_fpath = os.path.join(self.chkpt_path, "best_model.ckpt")
    saver.restore(sess, chkpt_fpath)
    tf.get_variable_scope().reuse_variables()
    return sess

  def run(self,
          pianorolls,
          masks=None,
          sample_steps=0,
          current_step=0,
          total_gibbs_steps=0,
          temperature=0.99):
    """Given input pianorolls, runs Gibbs sampling to fill in the rest.

    When total_gibbs_steps is 0, total_gibbs_steps is set to
    time * instruments.  If faster sampling is desired on the expanse of sample
    quality, total_gibbs_steps can be explicitly set to a lower number,
    possibly to the value of sample_steps if do not plan on stopping sample
    early to obtain intermediate results.

    This function can be used to return intermediate results by setting the
    sample_steps to when results should be returned and leaving
    total_gibbs_steps to be 0.

    To continue sampling from intermediate results, set current_step to the
    number of steps taken, and feed in the intermediate pianorolls.  Again
    leaving total_gibbs_steps as 0.

    Args:
      pianorolls: a 4D numpy array of shape (batch, time, pitch, instrument)
      masks: a 4D numpy array of the same shape as pianorolls, with 1s
          indicating mask out.  If is None, then the masks will be where have 1s
          where there are no notes, indicating to the model they should be
          filled in.
      sample_steps: an integer indicating the number of steps to sample in this
          call.  If set to 0, then it defaults to total_gibbs_steps.
      current_step: an integer indicating how many steps might have already
          sampled before.
      total_gibbs_steps: an integer indicating the total number of steps that
          a complete sampling procedure would take.
      temperature: a float indicating the temperature for sampling from softmax.

    Returns:
      A dictionary, consisting of "pianorolls" which is a 4D numpy array of
      the sampled results and "time_taken" which is the time taken in sampling.
    """
    outer_masks = masks
    if outer_masks is None:
      outer_masks = lib_sampling.CompletionMasker()(pianorolls)

    start_time = time.time()
    new_piece = self.sess.run(
        self.samples,
        feed_dict={
            self.placeholders["pianorolls"]: pianorolls,
            self.placeholders["outer_masks"]: outer_masks,
            self.placeholders["sample_steps"]: sample_steps,
            self.placeholders["total_gibbs_steps"]: total_gibbs_steps,
            self.placeholders["current_step"]: current_step,
            self.placeholders["temperature"]: temperature
        })

    label = "independent blocked gibbs"
    time_taken = (time.time() - start_time) / 60.0
    tf.logging.info("exit  %s (%.3fmin)" % (label, time_taken))
    return dict(pianorolls=new_piece, time_taken=time_taken)


def make_bernoulli_masks(shape, pm, outer_masks=1.):
  bb = shape[0]
  tt = shape[1]
  pp = shape[2]
  ii = shape[3]
  probs = tf.random_uniform([bb, tt, ii])
  masks = tf.tile(tf.to_float(tf.less(probs, pm))[:, :, None, :], [1, 1, pp, 1])
  return masks * outer_masks


def sample_with_temperature(logits, temperature):
  """Either argmax after softmax or random sample along the pitch axis.

  Args:
    logits: a Tensor of shape (batch, time, pitch, instrument).
    temperature: a float  0.0=argmax 1.0=random

  Returns:
    a Tensor of the same shape, with one_hots on the pitch dimension.
  """
  logits = tf.transpose(logits, [0, 1, 3, 2])
  pitch_range = tf.shape(logits)[-1]

  def sample_from_logits(logits):
    with tf.control_dependencies([tf.assert_greater(temperature, 0.0)]):
      logits = tf.identity(logits)
    reshaped_logits = (
        tf.reshape(logits, [-1, tf.shape(logits)[-1]]) / temperature)
    choices = tf.multinomial(reshaped_logits, 1)
    choices = tf.reshape(choices,
                         tf.shape(logits)[:logits.get_shape().ndims - 1])
    return choices

  choices = tf.cond(tf.equal(temperature, 0.0),
                    lambda: tf.argmax(tf.nn.softmax(logits), -1),
                    lambda: sample_from_logits(logits))
  samples_onehot = tf.one_hot(choices, pitch_range)
  return tf.transpose(samples_onehot, [0, 1, 3, 2])


def compute_mask_prob_from_yao_schedule(i, n, pmin=0.1, pmax=0.9, alpha=0.7):
  wat = (pmax - pmin) * i/ n
  return tf.maximum(pmin, pmax - wat / alpha)


def main(unused_argv):
  checkpoint_path = FLAGS.checkpoint
  sampler = CoconetSampleGraph(checkpoint_path)

  batch_size = 1
  decode_length = 4
  target_shape = [batch_size, decode_length, 46, 4]
  pianorolls = np.zeros(target_shape, dtype=np.float32)
  generated_piece = sampler.run(pianorolls, sample_steps=16, temperature=0.99)
  tf.logging.info("num of notes in piece %d", np.sum(generated_piece))

  tf.logging.info("Done.")


if __name__ == "__main__":
  tf.app.run()
