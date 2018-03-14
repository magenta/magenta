"""Runs OMR and outputs a Score or NoteSequence message."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

# internal imports
from absl import app
from absl import flags

from google.protobuf import text_format

from magenta.models.omr import conversions
from magenta.models.omr import engine
from magenta.models.omr.glyphs import saved_classifier_fn

from tensorflow.python.lib.io import file_io

FLAGS = flags.FLAGS

VALID_OUTPUT_TYPES = ['MusicXML', 'NoteSequence', 'Score']

# The name of the png file contents tensor.
PNG_CONTENTS_TENSOR = 'png_contents'

flags.DEFINE_string(
    'glyphs_saved_model', None,
    'Path to the patch-based glyph classifier saved model dir. Defaults to the'
    ' included KNN classifier.')
flags.DEFINE_string(
    'output', '/dev/stdout',
    'Path to write the output text-format or binary proto.')
flags.DEFINE_string(
    'output_type', 'Score',
    'Which output type to produce (Score or NoteSequence).')
flags.DEFINE_boolean('text_format', True, 'Whether the output is text format.')


def run(input_pngs, glyphs_saved_model=None, output_notesequence=False):
  """Runs OMR over a list of input images.

  Args:
    input_pngs: A list of PNG filenames to process.
    glyphs_saved_model: Optional saved model dir to override the included model.
    output_notesequence: Whether to return a NoteSequence, as opposed to a Score
        containing Pages with Glyphs.

  Returns:
    A NoteSequence message, or a Score message holding Pages for each input
        image (with their detected Glyphs).
  """
  return engine.OMREngine(
      saved_classifier_fn.build_classifier_fn(glyphs_saved_model)).run(
          input_pngs, output_notesequence=output_notesequence)


def main(argv):
  if FLAGS.output_type not in VALID_OUTPUT_TYPES:
    raise ValueError('output_type "%s" not in allowed types: %s' %
                     (FLAGS.output_type, VALID_OUTPUT_TYPES))

  # Exclude argv[0], which is the current binary.
  patterns = argv[1:]
  if not patterns:
    raise ValueError('PNG file glob(s) must be specified')
  input_paths = []
  for pattern in patterns:
    pattern_paths = file_io.get_matching_files(pattern)
    if not pattern_paths:
      raise ValueError('Pattern "%s" failed to match any files' % pattern)
    input_paths.extend(pattern_paths)

  start = time.time()
  output = run(
      input_paths,
      FLAGS.glyphs_saved_model,
      output_notesequence=FLAGS.output_type == 'NoteSequence')
  end = time.time()
  sys.stderr.write('OMR elapsed time: %.2f\n' % (end - start))

  if FLAGS.output_type == 'MusicXML':
    output_bytes = conversions.score_to_musicxml(output).encode('utf-8')
  else:
    if FLAGS.text_format:
      output_bytes = text_format.MessageToString(output).encode('utf-8')
    else:
      output_bytes = output.SerializeToString()
  file_io.write_string_to_file(FLAGS.output, output_bytes)


if __name__ == '__main__':
  app.run(main)
