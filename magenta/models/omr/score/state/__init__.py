"""The global state for the entire score."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from six import moves

from magenta.models.omr.score.state import staff as staff_state


class ScoreState(object):
  """The global state for the entire score.

  Represents the state of the score across multiple staff systems. The staff
  system state does not change on a new line, unless the number of staves
  changes. `num_staves` should be called for every new staff system, to reset
  the staff state properly.

  Attributes:
    staves: A list of StaffState objects, representing each staff in the current
        staff system.
  """

  def __init__(self):
    self.staves = []

  def num_staves(self, num_staves):
    """Updates the score to have the given number of staves.

    If `num_staves` matches the current `len(self.staves)`, copies the persisted
    state from the previous staves to the new staff system. Otherwise,
    discards any current staves and constructs `num_staves` new staves.

    Args:
      num_staves: The number of staves for the current staff system.
    """
    time = self.add_measure()
    if len(self.staves) != self.num_staves:
      self.staves = [
          staff_state.StaffState(time) for _ in moves.range(num_staves)
      ]
    else:
      self.staves = [staff.new_staff(time) for staff in self.staves]

  def add_measure(self):
    """Adds a new measure for the current staff system.

    Called on every bar. Updates each staff.

    Returns:
      The start time of the new measure, which is the max of the current time of
      each current staff.
    """
    time = (max([staff.get_time() for staff in self.staves])
            if self.staves else 0)
    for staff in self.staves:
      staff.add_measure(time)
    return time
