"""A naive horizontal projection-based staff detector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import scipy.ndimage
import tensorflow as tf

from magenta.models.omr.staves import base
from magenta.models.omr.util import memoize


class ProjectionStaffDetector(base.BaseStaffDetector):
  """A naive staff detector that uses horizontal projections.

  Detects peaks in the number of black pixels in each row, which should
  correspond to staff lines.
  """
  staves_tensor = None
  staffline_distance_tensor = None

  def __init__(self, image=None):
    super(ProjectionStaffDetector, self).__init__(image)
    projection = tf.reduce_sum(tf.cast(self.image <= 127, tf.int32), 1)
    width = tf.shape(self.image)[1]
    min_num_dark_pixels = width // 2
    staff_lines = projection > min_num_dark_pixels
    staves, staffline_distance, staffline_thickness = tf.py_func(
        _projection_to_staves, [staff_lines, width],
        [tf.int32, tf.int32, tf.int32])
    self.staves_tensor = staves
    self.staffline_distance_tensor = staffline_distance
    self.staffline_thickness_tensor = staffline_thickness

  @property
  @memoize.MemoizedFunction
  def staves(self):
    return self.staves_tensor

  @property
  @memoize.MemoizedFunction
  def staffline_distance(self):
    return self.staffline_distance_tensor

  @property
  @memoize.MemoizedFunction
  def staffline_thickness(self):
    return self.staffline_thickness_tensor


def _projection_to_staves(projection, width):
  """Pure python implementation of projection-based staff detection."""
  labels, num_labels = scipy.ndimage.measurements.label(projection)
  current_staff = []
  staff_center_lines = []
  staffline_distance = []
  staffline_thicknesses = []
  for line in range(1, num_labels + 1):
    line_start = np.where(labels == line)[0].min()
    line_end = np.where(labels == line)[0].max()
    staffline_thickness = line_end - line_start + 1
    line_center = np.int32(round((line_start + line_end) / 2.0))
    current_staff.append(line_center)
    if len(current_staff) > 5:
      del current_staff[0]
    if len(current_staff) == 5:
      dists = np.array([current_staff[1] - current_staff[0],
                        current_staff[2] - current_staff[1],
                        current_staff[3] - current_staff[2],
                        current_staff[4] - current_staff[3]])
      if np.max(dists) - np.min(dists) < 3:
        staff_center = round(np.mean(current_staff))
        staff_center_lines.append([[0, staff_center],
                                   [width - 1, staff_center]])
        staffline_distance.append(round(np.mean(dists)))
        staffline_thicknesses.append(staffline_thickness)
  staffline_thickness = (
      np.median(staffline_thicknesses).astype(np.int32)
      if staffline_thicknesses
      else np.int32(1))
  return (np.array(staff_center_lines, np.int32),
          np.array(staffline_distance, np.int32),
          staffline_thickness)
