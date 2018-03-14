"""Staff detection test utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.staves import base


class FakeStaves(base.BaseStaffDetector):
  """Fake staff detector holding an arbitrary staves tensor.

  Attributes:
    image: The image.
    staves_t: The staves given to the constructor. None may be given if the
        staves are never checked (only the staffline distance).
    staffline_distance_t: The estimated staffline distance. 1D tensor (values
        for each staff) or None.
    staffline_thickness_t: The estimated staffline thickness. Scalar tensor or
        None.
  """

  def __init__(self,
               image_t,
               staves_t,
               staffline_distance_t=None,
               staffline_thickness_t=None):
    self.image = image_t
    self.staves_t = staves_t
    self.staffline_distance_t = staffline_distance_t
    self.staffline_thickness_t = staffline_thickness_t

  @property
  def staves(self):
    return self.staves_t

  @property
  def staffline_distance(self):
    return self.staffline_distance_t

  @property
  def staffline_thickness(self):
    return self.staffline_thickness_t
