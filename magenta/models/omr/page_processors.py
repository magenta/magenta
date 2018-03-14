"""Callables that process Page messages.

Each processor takes a Page and returns a possibly modified Page. It may modify
the Page message in place, and return the same message.

The purpose of a processor is to perform simple inference on elements already in
the Page and in the Structure. Processing should not be CPU-intensive, or the
heavy lifting needs to be implemented in TensorFlow for efficiency.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.glyphs import glyph_types
from magenta.models.omr.glyphs import note_dots
from magenta.models.omr.glyphs import repeated
from magenta.models.omr.staves import staff_processor
from magenta.models.omr.structure import barlines
from magenta.models.omr.structure import beam_processor
from magenta.models.omr.structure import section_barlines
from magenta.models.omr.structure import stems


def create_processors(structure, staffline_extractor=None):
  """Generator for the processors to be applied to the Page in order.

  Args:
    structure: The computed `Structure`.
    staffline_extractor: The staffline extractor to use for scaling glyph x
      coordinates. Optional.

  Yields:
    Callables which accept a single `Page` as an argument, and return it
      (either modifying in place or returning a modified copy).
  """
  yield staff_processor.StaffProcessor(structure, staffline_extractor)
  yield stems.Stems(structure)
  yield beam_processor.BeamProcessor(structure)
  yield note_dots.NoteDots(structure)

  yield CenteredRests()
  yield repeated.FixRepeatedRests()

  yield barlines.Barlines(structure)
  yield section_barlines.SectionBarlines(structure)
  yield section_barlines.MergeStandardAndBeginRepeatBars(structure)


def process(page, structure, staffline_extractor=None):
  for processor in create_processors(structure, staffline_extractor):
    page = processor.apply(page)
  return page


# TODO(ringwalt): Add a helper for processors that filter the glyphs like this.
class CenteredRests(object):

  def apply(self, page):
    """Rests should be centered on the staff, assuming a single voice."""
    for system in page.system:
      for staff in system.staff:
        to_remove = []
        for glyph in staff.glyph:
          if glyph_types.is_rest(glyph) and abs(glyph.y_position) > 2:
            to_remove.append(glyph)

        for glyph in to_remove:
          staff.glyph.remove(glyph)

    return page
