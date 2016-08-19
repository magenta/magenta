"""Train teacher-forcing sequence models."""
import os
import tensorflow as tf

import magenta.models.wayback.lib.evaluation as evaluation
import magenta.models.wayback.lib.hyperparameters as hyperparameters
import magenta.models.wayback.lib.models as models
from magenta.models.wayback.lib.namespace import Namespace as NS
import magenta.models.wayback.lib.training as training
import magenta.models.wayback.lib.wavefile as wavefile

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("base_output_dir",
                       "/tmp/models",
                       "output directory where models should be stored")
tf.flags.DEFINE_bool("resume", False,
                     "resume training from a checkpoint or delete and restart")
tf.flags.DEFINE_integer("max_step_count", 100000,
                        "max number of training steps")
tf.flags.DEFINE_integer("max_examples", None, "number of examples to train on")
tf.flags.DEFINE_integer("validation_interval", 100,
                        "number of training steps between validations")
tf.flags.DEFINE_string("basename", None, "model name prefix")
tf.flags.DEFINE_string("hyperparameters", "", "hyperparameter settings")
tf.flags.DEFINE_string("data_dir", None, "path to data directory; must have"
                       " train/valid/test subdirectories containing wav files")


class StopTraining(Exception):
  pass


def get_model_name(hp):
  """Compile a name for the model based on hyperparameters.

  Args:
    hp: hyperparameter Namespace.

  Returns:
    Model name as a string.
  """
  fragments = []
  if FLAGS.basename:
    fragments.append(FLAGS.basename)
  fragments.append("sf%d" % hp.sampling_frequency)
  fragments.append("bd%d" % hp.bit_depth)
  fragments.extend([
      hp.layout, hp.cell,
      "s%d" % hp.segment_length,
      "c%d" % hp.chunk_size,
      "p%s" % ",".join(list(map(str, hp.periods))),
      "l%s" % ",".join(list(map(str, hp.layer_sizes))),
      "u%d" % hp.unroll_layer_count,
      "bn%s" % hp.use_bn,
      "a%s" % hp.activation,
      "bs%d" % hp.batch_size,
      "io%s" % ",".join(list(map(str, hp.io_sizes))),
      "carry%s" % hp.carry,
  ])
  return "_".join(fragments)


def main(argv):
  # ensure flags were parsed correctly
  assert not argv[1:]

  hp = hyperparameters.parse(FLAGS.hyperparameters)

  print "loading data from %s" % FLAGS.data_dir
  dataset = wavefile.Dataset(NS((fold,
                                 tf.gfile.Glob(os.path.join(FLAGS.data_dir,
                                                            "%s/*.wav" % fold)))
                                for fold in "train valid test".split()),
                             frequency=hp.sampling_frequency,
                             bit_depth=hp.bit_depth)
  print "done"
  hp.data_dim = dataset.data_dim

  model_name = get_model_name(hp)
  print model_name
  output_dir = os.path.join(FLAGS.base_output_dir, model_name)

  if not FLAGS.resume:
    if tf.gfile.Exists(output_dir):
      tf.gfile.DeleteRecursively(output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  hyperparameters.dump(os.path.join(output_dir, "hyperparameters.yaml"), hp)

  model = models.construct(hp)

  print "constructing graph..."
  global_step = tf.Variable(0, trainable=False, name="global_step")
  trainer = training.Trainer(model, hp=hp, global_step=global_step)
  tf.get_variable_scope().reuse_variables()
  evaluator = evaluation.Evaluator(model, hp=hp)
  print "done"

  best_saver = tf.Saver()
  supervisor = tf.Supervisor(logdir=output_dir, summary_op=None)
  session = supervisor.PrepareSession("local")
  try:
    tracking = NS(best_loss=None, reset_time=0)

    def maybe_validate(state):
      if state.global_step % FLAGS.validation_interval == 0:
        values = evaluator.run(
            examples=dataset.examples.valid, session=session, hp=hp,
            # don't spend too much time evaluating
            max_step_count=FLAGS.validation_interval // 3)

        supervisor.summary_computed(
            session, tf.Summary(value=values.summaries))

        if tracking.best_loss is None or values.loss < tracking.best_loss:
          tracking.best_loss = values.loss
          tracking.reset_time = state.global_step
          best_saver.save(
              session,
              os.path.join(os.path.dirname(supervisor.save_path),
                           "best_%i_%s.ckpt"
                           % (state.global_step, values.loss)),
              global_step=supervisor.global_step)

        elif state.global_step - tracking.reset_time > hp.decay_patience:
          session.run(trainer.tensors.decay_op)
          tracking.reset_time = state.global_step

    def maybe_stop(_):
      if supervisor.ShouldStop():
        raise StopTraining()

    def before_step_hook(state):
      maybe_validate(state)
      maybe_stop(state)

    def after_step_hook(_, values):
      for summary in values.summaries:
        supervisor.summary_computed(session, summary)

    print "training."
    try:
      trainer.run(examples=dataset.examples.train[:FLAGS.max_examples],
                  session=session, hp=hp, max_step_count=FLAGS.max_step_count,
                  hooks=NS(step=NS(before=before_step_hook,
                                   after=after_step_hook)))
    except StopTraining:
      pass
  finally:
    supervisor.Stop()


if __name__ == "__main__":
  tf.app.run()
