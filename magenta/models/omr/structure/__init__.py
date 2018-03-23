"""Holder for the page structure detectors.

`create_structure()` constructs the staff and verticals detectors with the
given callables. `Structure.compute()` is run to compute all structure in a
single TensorFlow graph, to increase parallelism.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from magenta.models.omr import staves
from magenta.models.omr.staves import base as staves_base
from magenta.models.omr.staves import removal
from magenta.models.omr.structure import beams as beams_module
from magenta.models.omr.structure import components as components_module
from magenta.models.omr.structure import verticals as verticals_module


def create_structure(image,
                     staff_detector=staves.StaffDetector,
                     beams=beams_module.Beams,
                     verticals=verticals_module.ColumnBasedVerticals,
                     components=components_module.from_staff_remover):
  """Constructs a Structure instance.

  Constructs a staff detector and verticals with the given callables.

  Args:
    image: The image tensor.
    staff_detector: A callable that accepts the image and returns a
        StaffDetector.
    beams: A callable that accept a StaffRemover and returns a Beams.
    verticals: A callable that accepts the staff detector and returns a
        verticals impl (e.g. ColumnBasedVerticals).
    components: A callable that accepts a StaffRemover and returns a
        ConnectedComponents.

  Returns:
    The Structure instance.
  """
  with tf.name_scope('staff_detector'):
    staff_detector = staff_detector(image)
  with tf.name_scope('staff_remover'):
    staff_remover = removal.StaffRemover(staff_detector)
  with tf.name_scope('beams'):
    beams = beams(staff_remover)
  with tf.name_scope('verticals'):
    verticals = verticals(staff_detector)
  with tf.name_scope('components'):
    components = components(staff_remover)
  structure = Structure(
      staff_detector,
      beams,
      verticals,
      components,
      image=image,
      staff_remover=staff_remover)
  return structure


class Structure(object):
  """Holds page structure detectors."""

  def __init__(self,
               staff_detector,
               beams,
               verticals,
               connected_components,
               image=None,
               staff_remover=None):
    self.image = image
    self.staff_detector = staff_detector
    self.beams = beams
    self.verticals = verticals
    self.connected_components = connected_components
    self.staff_remover = staff_remover

  def compute(self, session=None, image=None):
    """Computes the structure.

    If the staves are already `ComputedStaves` and the verticals are already
    `ComputedVerticals`, returns `self`. Otherwise, runs staff detection and/or
    verticals detection in the TensorFlow `session`.

    Args:
      session: The TensorFlow session to use instead of the default session.
      image: If non-None, fed as the value of `self.staff_detector.image`.

    Returns:
      A computed `Structure` object. `staff_detector` and `verticals` hold NumPy
          arrays with the result of the TensorFlow graph.
    """

    if isinstance(self.staff_detector, staves_base.ComputedStaves):
      staff_detector_data = []
    else:
      staff_detector_data = self.staff_detector.data
    if isinstance(self.beams, beams_module.ComputedBeams):
      beams_data = []
    else:
      beams_data = self.beams.data
    if isinstance(self.verticals, verticals_module.ComputedVerticals):
      verticals_data = []
    else:
      verticals_data = self.verticals.data
    if isinstance(self.connected_components,
                  components_module.ComputedComponents):
      components_data = []
    else:
      components_data = self.connected_components.data
    if not (staff_detector_data or beams_data or verticals_data or
            components_data):
      return self

    if not session:
      session = tf.get_default_session()
    if image is not None:
      feed_dict = {self.staff_detector.image: image}
    else:
      feed_dict = {}
    staff_detector_data, beams_data, verticals_data, components_data = (
        session.run(
            [staff_detector_data, beams_data, verticals_data, components_data],
            feed_dict=feed_dict))
    staff_detector_data = staff_detector_data or self.staff_detector.data
    staff_detector = staves_base.ComputedStaves(*staff_detector_data)
    beams_data = beams_data or self.beams.data
    beams = beams_module.ComputedBeams(*beams_data)
    verticals_data = verticals_data or self.verticals.data
    verticals = verticals_module.ComputedVerticals(*verticals_data)
    connected_components = components_module.ConnectedComponents(
        *components_data)
    return Structure(
        staff_detector, beams, verticals, connected_components, image=image)

  def is_computed(self):
    return (isinstance(self.staff_detector, staves_base.ComputedStaves) and
            isinstance(self.beams, beams_module.ComputedBeams) and
            isinstance(self.verticals, verticals_module.ComputedVerticals) and
            isinstance(self.connected_components,
                       components_module.ComputedComponents))

  @property
  def data(self):
    return [
        self.staff_detector.data, self.beams.data, self.verticals.data,
        self.connected_components.data
    ]
