"""Connected component analysis.

Connected components will be used to detect solid-ish elements or blobs on the
score (e.g. beams, dots, and whole/half rests).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import enum
import tensorflow as tf


class ConnectedComponentsColumns(enum.IntEnum):
  """The field names of the connected components 2D array columns."""

  X0 = 0
  Y0 = 1
  X1 = 2
  Y1 = 3
  SIZE = 4


def from_staff_remover(staff_remover, threshold=127):
  with tf.name_scope('from_staff_remover'):
    image = tf.less(staff_remover.remove_staves, threshold)
    return ConnectedComponents(get_component_bounds(image))


class ConnectedComponents(object):
  """Holds the connected components on the staves removed image."""

  def __init__(self, components):
    with tf.name_scope('ConnectedComponents'):
      self.components = components
      self.data = [self.components]


class ComputedComponents(object):
  """Holds the computed NumPy array of connected components."""

  def __init__(self, components):
    self.components = components
    self.data = [components]


def get_component_bounds(image):
  """Returns the bounding box of each connected component in `image`.

  Connected components are segments of adjacent True pixels in the image.

  Args:
    image: A 2D boolean image tensor.

  Returns:
    A tensor of shape (num_components, 5), where each row represents a connected
        component of the image as `(x0, y0, x1, y1, size)`. `size` is the count
        of True pixels in the component, and the coordinates are the top left
        and bottom right corners of the bounding box.
  """
  with tf.name_scope('get_component_bounds'):
    components = tf.contrib.image.connected_components(image)
    num_components = tf.reduce_max(components) + 1
    width = tf.shape(image)[1]
    height = tf.shape(image)[0]
    xs, ys = tf.meshgrid(tf.range(width), tf.range(height))
    component_x0 = _unsorted_segment_min(xs, components, num_components)[1:]
    component_x1 = tf.unsorted_segment_max(xs, components, num_components)[1:]
    component_y0 = _unsorted_segment_min(ys, components, num_components)[1:]
    component_y1 = tf.unsorted_segment_max(ys, components, num_components)[1:]
    component_size = tf.bincount(components)[1:]
    return tf.stack(
        [
            component_x0, component_y0, component_x1, component_y1,
            component_size
        ],
        axis=1)


def _unsorted_segment_min(data, segment_ids, num_segments):
  return -tf.unsorted_segment_max(-data, segment_ids, num_segments)
