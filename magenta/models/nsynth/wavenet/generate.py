import os
import tensorflow as tf
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.fastgen import synthesize

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("source_path", "", "Path to directory with either "
													 ".wav files or precomputed encodings in .npy files.")
tf.app.flags.DEFINE_string("encodings", False, "Path to npy files.")
tf.app.flags.DEFINE_string("save_path", "", "Path to output file dir.")
tf.app.flags.DEFINE_string("checkpoint_path", "model.ckpt-200000", "Path to checkpoint.")
tf.app.flags.DEFINE_integer("sample_length", 64000, "Output file size in samples.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


def main(unused_argv=None):
  source_path = utils.shell_path(FLAGS.source_path)
  encoding_path = utils.shell_path(FLAGS.encoding_path)
  checkpoint_path = utils.shell_path(FLAGS.checkpoint_path)
  save_path = utils.shell_path(FLAGS.save_path)
  if not save_path:
  	raise RuntimeError("Must specify a save_path.")
  tf.logging.set_verbosity(FLAGS.log)
  
  # Generate from wav files
  if source_path:
  	if tf.gfile.IsDirectory(source_path):
	  wavfiles = sorted([os.path.join(source_path, fname) 
	  					 for fname in tf.gfile.ListDirectory(source_path)
	                     if fname.lower().endswith(".wav")])
	elif source_path.lower().endswith(".wav"):
	  wavfiles = [source_path]
	else:
	  wavfiles = []
	for wav_file in wavfiles:
	  out_file = os.path.join(save_path,
	  						  "gen_" + os.path.basename(wav_file))
	  tf.logging.info("OUTFILE %s" % out_file)
	  synthesize(wav_file=wav_file,
	             checkpoint_path=checkpoint_path,
	             out_file=out_file,
	             sample_length=FLAGS.sample_length)
	# Or generate from precomputed embeddings
  elif encoding_path:
  	if tf.gfile.IsDirectory(encoding_path):
	  embedding_files = sorted([os.path.join(encoding_path, fname) 
	  					 for fname in tf.gfile.ListDirectory(encoding_path)
	                     if fname.lower().endswith(".npy")])
	elif encoding_path.lower().endswith(".npy"):
	  embedding_files = [encoding_path]
	else:
	  embedding_files = []
	for wav_file in embedding_files:
	  out_file = os.path.join(save_path,
	  						  "gen_" + os.path.basename(wav_file))
	  tf.logging.info("OUTFILE %s" % out_file)
	  synthesize(encoding_file=encoding_file,
	             checkpoint_path=checkpoint_path,
	             out_file=out_file,
	             sample_length=FLAGS.sample_length)


if __name__ == "__main__":
  tf.app.run()
