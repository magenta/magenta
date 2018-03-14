"""Tool to convert existing KNN tfrecords to a saved model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from absl import app
import tensorflow as tf

from magenta.models.omr.glyphs import corpus
from magenta.models.omr.glyphs import knn_model


def run(tfrecords_filename, export_dir):
  with tf.Session():
    height, width = corpus.get_patch_shape(tfrecords_filename)
    patches, labels = corpus.parse_corpus(tfrecords_filename, height, width)

  knn_model.export_knn_model(patches, labels, export_dir)


def main(argv):
  _, infile, outdir = argv
  run(infile, outdir)


if __name__ == '__main__':
  app.run(main)
