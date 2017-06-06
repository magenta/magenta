import tensorflow as tf
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.fastgen import synthesize

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("wav_file", "input.wav", "Path to input file.")
tf.app.flags.DEFINE_string("out_file",  "synthesis.wav", "Path to output file.")
tf.app.flags.DEFINE_string("ckpt_path", "model.ckpt-200000", "Path to checkpoint.")
tf.app.flags.DEFINE_integer("sample_length", 64000, "Output file size in samples.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


def main(unused_argv=None):
  wav_file = utils.shell_path(FLAGS.wav_file)
  ckpt_path = utils.shell_path(FLAGS.ckpt_path)
  out_file = utils.shell_path(FLAGS.out_file)
  tf.logging.set_verbosity(FLAGS.log)
  synthesize(wav_file=wav_file,
             ckpt_path=ckpt_path,
             out_file=out_file,
             sample_length=FLAGS.sample_length)

if __name__ == "__main__":
  tf.app.run()
