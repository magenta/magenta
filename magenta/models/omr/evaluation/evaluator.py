"""Evaluates OMR using ground truth images and MusicXML."""
# TODO(ringwalt): Maybe add a flag for exporting CSV. We might also want to join
# all DataFrames, with a new index for the ground truth title, if that's not too
# unwieldy.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from google.protobuf import text_format

from magenta.models.omr import conversions
from magenta.models.omr import engine
from magenta.models.omr.evaluation import musicxml
from magenta.models.omr.protobuf import groundtruth_pb2

from tensorflow.python.lib.io import file_io


class Evaluator(object):

  def __init__(self, **kwargs):
    self.omr = engine.OMREngine(**kwargs)

  def evaluate(self, ground_truth):
    expected = (
        file_io.read_file_to_string(ground_truth.ground_truth_filename)
        .encode('utf-8'))
    score = self.omr.run(
        page_spec.filename for page_spec in ground_truth.page_spec)
    actual = conversions.score_to_musicxml(score)
    return musicxml.musicxml_similarity(actual, expected)


def main(argv):
  if len(argv) <= 1:
    raise ValueError('Ground truth filenames are required')
  evaluator = Evaluator()
  for ground_truth_file in argv[1:]:
    truth = groundtruth_pb2.GroundTruth()
    text_format.Parse(file_io.read_file_to_string(ground_truth_file), truth)
    print(truth.title)
    print(evaluator.evaluate(truth))


if __name__ == '__main__':
  tf.app.run()
