# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Generate from trained model from scratch or condition on a partial score."""
import itertools as it
import os
import re
import time

from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_logging
from magenta.models.coconet import lib_mask
from magenta.models.coconet import lib_pianoroll
from magenta.models.coconet import lib_sampling
from magenta.models.coconet import lib_tfsampling
from magenta.models.coconet import lib_util
import numpy as np
import pretty_midi
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_integer("gen_batch_size", 3,
                     "Num of samples to generate in a batch.")
flags.DEFINE_string("strategy", None,
                    "Use complete_midi when using midi.")
flags.DEFINE_float("temperature", 0.99, "Softmax temperature")
flags.DEFINE_integer("piece_length", 32, "Num of time steps in generated piece")
flags.DEFINE_string("generation_output_dir", None,
                    "Output directory for storing the generated Midi.")
flags.DEFINE_string("prime_midi_melody_fpath", None,
                    "Path to midi melody to be harmonized.")
flags.DEFINE_string("checkpoint", None, "Path to checkpoint file")
flags.DEFINE_bool("midi_io", False, "Run in midi in and midi out mode."
                  "Does not write any midi or logs to disk.")
flags.DEFINE_bool("tfsample", True, "Run sampling in Tensorflow graph.")


def main(unused_argv):
  if FLAGS.checkpoint is None or not FLAGS.checkpoint:
    raise ValueError(
        "Need to provide a path to checkpoint directory.")

  if FLAGS.tfsample:
    generator = TFGenerator(FLAGS.checkpoint)
  else:
    wmodel = instantiate_model(FLAGS.checkpoint)
    generator = Generator(wmodel, FLAGS.strategy)
  midi_outs = generator.run_generation(
      gen_batch_size=FLAGS.gen_batch_size, piece_length=FLAGS.piece_length)

  # Creates a folder for storing the process of the sampling.
  label = "sample_%s_%s_%s_T%g_l%i_%.2fmin" % (lib_util.timestamp(),
                                               FLAGS.strategy,
                                               generator.hparams.architecture,
                                               FLAGS.temperature,
                                               FLAGS.piece_length,
                                               generator.time_taken)
  basepath = os.path.join(FLAGS.generation_output_dir, label)
  tf.logging.info("basepath: %s", basepath)
  tf.gfile.MakeDirs(basepath)

  # Saves the results as midi or returns as midi out.
  midi_path = os.path.join(basepath, "midi")
  tf.gfile.MakeDirs(midi_path)
  tf.logging.info("Made directory %s", midi_path)
  save_midis(midi_outs, midi_path, label)

  result_npy_save_path = os.path.join(basepath, "generated_result.npy")
  tf.logging.info("Writing final result to %s", result_npy_save_path)
  with tf.gfile.Open(result_npy_save_path, "w") as p:
    np.save(p, generator.pianorolls)

  if FLAGS.tfsample:
    tf.logging.info("Done")
    return

  # Stores all the (intermediate) steps.
  intermediate_steps_path = os.path.join(basepath, "intermediate_steps.npz")
  with lib_util.timing("writing_out_sample_npz"):
    tf.logging.info("Writing intermediate steps to %s", intermediate_steps_path)
    generator.logger.dump(intermediate_steps_path)

  # Save the prime as midi and npy if in harmonization mode.
  # First, checks the stored npz for the first (context) and last step.
  tf.logging.info("Reading to check %s", intermediate_steps_path)
  with tf.gfile.Open(intermediate_steps_path, "r") as p:
    foo = np.load(p)
    for key in foo.keys():
      if re.match(r"0_root/.*?_strategy/.*?_context/0_pianorolls", key):
        context_rolls = foo[key]
        context_fpath = os.path.join(basepath, "context.npy")
        tf.logging.info("Writing context to %s", context_fpath)
        with lib_util.atomic_file(context_fpath) as context_p:
          np.save(context_p, context_rolls)
        if "harm" in FLAGS.strategy:
          # Only synthesize the one prime if in Midi-melody-prime mode.
          primes = context_rolls
          if "Melody" in FLAGS.strategy:
            primes = [context_rolls[0]]
          prime_midi_outs = get_midi_from_pianorolls(primes, generator.decoder)
          save_midis(prime_midi_outs, midi_path, label + "_prime")
        break
  tf.logging.info("Done")


class Generator(object):
  """Instantiates model and generates according to strategy and midi input."""

  def __init__(self, wmodel, strategy_name="complete_midi"):
    """Initializes Generator with a wrapped model and strategy name.

    Args:
      wmodel: A lib_tfutil.WrappedModel loaded from a model checkpoint.
      strategy_name: A string specifying the key of the default strategy.
    """
    self.wmodel = wmodel
    self.hparams = self.wmodel.hparams
    self.decoder = lib_pianoroll.get_pianoroll_encoder_decoder(self.hparams)
    self.logger = lib_logging.Logger()
    # Instantiates generation strategy.
    self.strategy_name = strategy_name
    self.strategy = BaseStrategy.make(self.strategy_name, self.wmodel,
                                      self.logger, self.decoder)
    self._pianorolls = None
    self._time_taken = None

  def run_generation(self,
                     midi_in=None,
                     pianorolls_in=None,
                     gen_batch_size=3,
                     piece_length=16,
                     new_strategy=None):
    """Generates, conditions on midi_in if given, returns midi.

    Args:
      midi_in: An optional PrettyMIDI instance containing notes to be
          conditioned on.
      pianorolls_in: An optional numpy.ndarray encoding the notes to be
          conditioned on as pianorolls.
      gen_batch_size: An integer specifying the number of outputs to generate.
      piece_length: An integer specifying the desired number of time steps to
          generate for the output, where a time step corresponds to the
          shortest duration supported by the model.
      new_strategy: new_strategy: A string specifying the key of the strategy
          to use. If not set, the most recently set strategy is used. If a
          strategy was never specified, then the default strategy that was
          instantiated during initialization is used.

    Returns:
      A list of PrettyMIDI instances, with length gen_batch_size.
    """
    if new_strategy is not None:
      self.strategy_name = new_strategy
      self.strategy = BaseStrategy.make(self.strategy_name, self.wmodel,
                                        self.logger, self.decoder)
    # Update the length of piece to be generated.
    self.hparams.crop_piece_len = piece_length
    shape = [gen_batch_size] + self.hparams.pianoroll_shape
    tf.logging.info("Tentative shape of pianorolls to be generated: %r", shape)

    # Generates.
    start_time = time.time()

    if midi_in is not None and "midi" in self.strategy_name.lower():
      pianorolls = self.strategy((shape, midi_in))
    elif "complete_manual" == self.strategy_name.lower():
      pianorolls = self.strategy(pianorolls_in)
    else:
      pianorolls = self.strategy(shape)
    self._pianorolls = pianorolls
    self._time_taken = (time.time() - start_time) / 60.0

    # Logs final step
    self.logger.log(pianorolls=pianorolls)

    midi_outs = get_midi_from_pianorolls(pianorolls, self.decoder)
    return midi_outs

  @property
  def pianorolls(self):
    return self._pianorolls

  @property
  def time_taken(self):
    return self._time_taken


class TFGenerator(object):
  """Instantiates model and generates according to strategy and midi input."""

  def __init__(self, checkpoint_path):
    """Initializes Generator with a wrapped model and strategy name.

    Args:
      checkpoint_path: A string that gives the full path to the folder that
          holds the checkpoint.
    """
    self.sampler = lib_tfsampling.CoconetSampleGraph(checkpoint_path)
    self.hparams = self.sampler.hparams
    self.endecoder = lib_pianoroll.get_pianoroll_encoder_decoder(self.hparams)

    self._time_taken = None
    self._pianorolls = None

  def run_generation(self,
                     midi_in=None,
                     pianorolls_in=None,
                     masks=None,
                     gen_batch_size=3,
                     piece_length=16,
                     sample_steps=0,
                     current_step=0,
                     total_gibbs_steps=0,
                     temperature=0.99):
    """Generates, conditions on midi_in if given, returns midi.

    Args:
      midi_in: An optional PrettyMIDI instance containing notes to be
          conditioned on.
      pianorolls_in: An optional numpy.ndarray encoding the notes to be
          conditioned on as pianorolls.
      masks: a 4D numpy array of the same shape as pianorolls, with 1s
          indicating mask out.  If is None, then the masks will be where have 1s
          where there are no notes, indicating to the model they should be
          filled in.
      gen_batch_size: An integer specifying the number of outputs to generate.
      piece_length: An integer specifying the desired number of time steps to
          generate for the output, where a time step corresponds to the
          shortest duration supported by the model.
      See the CoconetSampleGraph class in lib_tfsample.py for more detail.
      sample_steps: an integer indicating the number of steps to sample in this
          call.  If set to 0, then it defaults to total_gibbs_steps.
      current_step: an integer indicating how many steps might have already
          sampled before.
      total_gibbs_steps: an integer indicating the total number of steps that
          a complete sampling procedure would take.
      temperature: a float indicating the temperature for sampling from softmax.

    Returns:
      A list of PrettyMIDI instances, with length gen_batch_size.
    """
    # Update the length of piece to be generated.
    self.hparams.crop_piece_len = piece_length
    shape = [gen_batch_size] + self.hparams.pianoroll_shape
    tf.logging.info("Tentative shape of pianorolls to be generated: %r", shape)

    # Generates.
    if midi_in is not None:
      pianorolls_in = self.endecoder.encode_midi_to_pianoroll(midi_in, shape)
    elif pianorolls_in is None:
      pianorolls_in = np.zeros(shape, dtype=np.float32)
    results = self.sampler.run(
        pianorolls_in, masks, sample_steps=sample_steps,
        current_step=current_step, total_gibbs_steps=total_gibbs_steps,
        temperature=temperature)
    self._pianorolls = results["pianorolls"]
    self._time_taken = results["time_taken"]
    tf.logging.info("output pianorolls shape: %r", self.pianorolls.shape)
    midi_outs = get_midi_from_pianorolls(self.pianorolls, self.endecoder)
    return midi_outs

  @property
  def pianorolls(self):
    return self._pianorolls

  @property
  def time_taken(self):
    return self._time_taken


def get_midi_from_pianorolls(rolls, decoder):
  midi_datas = []
  for pianoroll in rolls:
    tf.logging.info("pianoroll shape: %r", pianoroll.shape)
    midi_data = decoder.decode_to_midi(pianoroll)
    midi_datas.append(midi_data)
  return midi_datas


def save_midis(midi_datas, midi_path, label=""):
  for i, midi_data in enumerate(midi_datas):
    midi_fpath = os.path.join(midi_path, "%s_%i.midi" % (label, i))
    tf.logging.info("Writing midi to %s", midi_fpath)
    with lib_util.atomic_file(midi_fpath) as p:
      midi_data.write(p)
  return midi_fpath


def instantiate_model(checkpoint, instantiate_sess=True):
  wmodel = lib_graph.load_checkpoint(
      checkpoint, instantiate_sess=instantiate_sess)
  return wmodel


##################
### Strategies ###
##################
# Commonly used compositions of samplers, user-selectable through FLAGS.strategy


class BaseStrategy(lib_util.Factory):
  """Base class for setting up generation strategies."""

  def __init__(self, wmodel, logger, decoder):
    self.wmodel = wmodel
    self.logger = logger
    self.decoder = decoder

  def __call__(self, shape):
    label = "%s_strategy" % self.key
    with lib_util.timing(label):
      with self.logger.section(label):
        return self.run(shape)

  def blank_slate(self, shape):
    return (np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32))

  # convenience function to avoid passing the same arguments over and over
  def make_sampler(self, key, **kwargs):
    kwargs.update(wmodel=self.wmodel, logger=self.logger)
    return lib_sampling.BaseSampler.make(key, **kwargs)


# pylint:disable=missing-docstring
class HarmonizeMidiMelodyStrategy(BaseStrategy):
  """Harmonizes a midi melody (fname given by FLAGS.prime_midi_melody_fpath)."""
  key = "harmonize_midi_melody"

  def load_midi_melody(self, midi=None):
    if midi is None:
      midi = pretty_midi.PrettyMIDI(FLAGS.prime_midi_melody_fpath)
    return self.decoder.encode_midi_melody_to_pianoroll(midi)

  def make_pianoroll_from_melody_roll(self, mroll, requested_shape):
    # mroll shape: time, pitch
    # requested_shape: batch, time, pitch, instrument
    bb, tt, pp, ii = requested_shape
    tf.logging.info("requested_shape: %r", requested_shape)
    assert mroll.ndim == 2
    assert mroll.shape[1] == 128
    hparams = self.wmodel.hparams
    assert pp == hparams.num_pitches, "%r != %r" % (pp, hparams.num_pitches)
    if tt != mroll.shape[0]:
      tf.logging.info("WARNING: requested tt %r != prime tt %r" % (
          tt, mroll.shape[0]))
    rolls = np.zeros((bb, mroll.shape[0], pp, ii), dtype=np.float32)
    rolls[:, :, :, 0] = mroll[None, :, hparams.min_pitch:hparams.max_pitch + 1]
    tf.logging.info("resulting shape: %r", rolls.shape)
    return rolls

  def run(self, tuple_in):
    shape, midi_in = tuple_in
    mroll = self.load_midi_melody(midi_in)
    pianorolls = self.make_pianoroll_from_melody_roll(mroll, shape)
    masks = lib_sampling.HarmonizationMasker()(pianorolls.shape)
    gibbs = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent", temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())

    with self.logger.section("context"):
      context = np.array([
          lib_mask.apply_mask(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)
      ])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)

    return pianorolls


class ScratchUpsamplingStrategy(BaseStrategy):
  key = "scratch_upsampling"

  def run(self, shape):
    # start with an empty pianoroll of length 1, then repeatedly upsample
    initial_shape = list(shape)
    desired_length = shape[1]
    initial_shape[1] = 1
    initial_shape = tuple(shape)

    pianorolls, masks = self.blank_slate(initial_shape)

    sampler = self.make_sampler(
        "upsampling",
        desired_length=desired_length,
        sampler=self.make_sampler(
            "gibbs",
            masker=lib_sampling.BernoulliMasker(),
            sampler=self.make_sampler(
                "independent", temperature=FLAGS.temperature),
            schedule=lib_sampling.YaoSchedule()))

    return sampler(pianorolls, masks)


class BachUpsamplingStrategy(BaseStrategy):
  key = "bach_upsampling"

  def run(self, shape):
    # optionally start with bach samples
    init_sampler = self.make_sampler("bach", temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)
    desired_length = 4 * shape[1]
    sampler = self.make_sampler(
        "upsampling",
        desired_length=desired_length,
        sampler=self.make_sampler(
            "gibbs",
            masker=lib_sampling.BernoulliMasker(),
            sampler=self.make_sampler(
                "independent", temperature=FLAGS.temperature),
            schedule=lib_sampling.YaoSchedule()))
    return sampler(pianorolls, masks)


class RevoiceStrategy(BaseStrategy):
  key = "revoice"

  def run(self, shape):
    init_sampler = self.make_sampler("bach", temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)

    sampler = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent", temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())

    for i in range(shape[-1]):
      masks = lib_sampling.InstrumentMasker(instrument=i)(shape)
      with self.logger.section("context"):
        context = np.array([
            lib_mask.apply_mask(pianoroll, mask)
            for pianoroll, mask in zip(pianorolls, masks)
        ])
        self.logger.log(pianorolls=context, masks=masks, predictions=context)
      pianorolls = sampler(pianorolls, masks)

    return pianorolls


class HarmonizationStrategy(BaseStrategy):
  key = "harmonization"

  def run(self, shape):
    init_sampler = self.make_sampler("bach", temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)

    masks = lib_sampling.HarmonizationMasker()(shape)

    gibbs = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent", temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())

    with self.logger.section("context"):
      context = np.array([
          lib_mask.apply_mask(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)
      ])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    with self.logger.section("result"):
      self.logger.log(
          pianorolls=pianorolls, masks=masks, predictions=pianorolls)

    return pianorolls


class TransitionStrategy(BaseStrategy):
  key = "transition"

  def run(self, shape):
    init_sampler = lib_sampling.BachSampler(
        wmodel=self.wmodel, temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)

    masks = lib_sampling.TransitionMasker()(shape)
    gibbs = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent", temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())

    with self.logger.section("context"):
      context = np.array([
          lib_mask.apply_mask(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)
      ])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    return pianorolls


class ChronologicalStrategy(BaseStrategy):
  key = "chronological"

  def run(self, shape):
    sampler = self.make_sampler(
        "ancestral",
        temperature=FLAGS.temperature,
        selector=lib_sampling.ChronologicalSelector())
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = sampler(pianorolls, masks)
    return pianorolls


class OrderlessStrategy(BaseStrategy):
  key = "orderless"

  def run(self, shape):
    sampler = self.make_sampler(
        "ancestral",
        temperature=FLAGS.temperature,
        selector=lib_sampling.OrderlessSelector())
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = sampler(pianorolls, masks)
    return pianorolls


class IgibbsStrategy(BaseStrategy):
  key = "igibbs"

  def run(self, shape):
    pianorolls, masks = self.blank_slate(shape)
    sampler = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent", temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())
    pianorolls = sampler(pianorolls, masks)
    return pianorolls


class AgibbsStrategy(BaseStrategy):
  key = "agibbs"

  def run(self, shape):
    pianorolls, masks = self.blank_slate(shape)
    sampler = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler(
            "ancestral",
            selector=lib_sampling.OrderlessSelector(),
            temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())
    pianorolls = sampler(pianorolls, masks)
    return pianorolls


class CompleteManualStrategy(BaseStrategy):
  key = "complete_manual"

  def run(self, pianorolls):
    # fill in the silences
    masks = lib_sampling.CompletionMasker()(pianorolls)
    gibbs = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent", temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())

    with self.logger.section("context"):
      context = np.array([
          lib_mask.apply_mask(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)
      ])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    with self.logger.section("result"):
      self.logger.log(
          pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls


class CompleteMidiStrategy(BaseStrategy):
  key = "complete_midi"

  def run(self, tuple_in):
    shape, midi_in = tuple_in
    pianorolls = self.decoder.encode_midi_to_pianoroll(midi_in, shape)
    # fill in the silences
    masks = lib_sampling.CompletionMasker()(pianorolls)
    gibbs = self.make_sampler(
        "gibbs",
        masker=lib_sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent", temperature=FLAGS.temperature),
        schedule=lib_sampling.YaoSchedule())

    with self.logger.section("context"):
      context = np.array([
          lib_mask.apply_mask(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)
      ])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    with self.logger.section("result"):
      self.logger.log(
          pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls

# pylint:enable=missing-docstring


# ok something else entirely.
def parse_art_to_pianoroll(art, tt=None):
  """Parse ascii art for pianoroll."""
  assert tt is not None
  ii = 4
  # TODO(annahuang): Properties of the model/data_tools, not of the ascii art.
  pmin, pmax = 36, 81
  pp = pmax - pmin + 1

  pianoroll = np.zeros((tt, pp, ii), dtype=np.float32)

  lines = art.strip().splitlines()
  klasses = "cCdDefFgGaAb"
  klass = None
  octave = None
  cycle = None
  for li, line in enumerate(lines):
    match = re.match(r"^\s*(?P<class>[a-gA-G])?(?P<octave>[0-9]|10)?\s*\|"
                     r"(?P<grid>[SATB +-]*)\|\s*$", line)
    if not match:
      if cycle is not None:
        print("ignoring unmatched line", li, repr(line))
      continue

    if cycle is None:
      # set up cycle through pitches and octaves
      print(match.groupdict())
      assert match.group("class") and match.group("class") in klasses
      assert match.group("octave")
      klass = match.group("class")
      octave = int(match.group("octave"))
      cycle = reversed(list(it.product(list(range(octave + 1)), klasses)))
      cycle = list(cycle)
      print(cycle)
      cycle = it.dropwhile(lambda ok: ok[1] != match.group("class"), cycle)  # pylint: disable=cell-var-from-loop
      o, k = next(cycle)
      assert k == klass
      assert o == octave
      cycle = list(cycle)
      print(cycle)
      cycle = iter(cycle)
    else:
      octave, klass = next(cycle)
      if match.group("class"):
        assert klass == match.group("class")
      if match.group("octave"):
        assert octave == int(match.group("octave"))

    pitch = octave * len(klasses) + klasses.index(klass)
    print(klass, octave, pitch, "\t", line)

    p = pitch - pmin
    for t, c in enumerate(match.group("grid")):
      if c in "+- ":
        continue
      i = "SATB".index(c)
      pianoroll[t, p, i] = 1.

  return pianoroll


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()
