"""The state for a single staff of a staff system."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.score.elements import clef as clef_module
from magenta.models.omr.score.state import measure


class StaffState(object):
  """The state for a single staff of a staff system.

  Holds the current measure of the staff, which has per-measure state (e.g.
  accidentals). Other state is copied to a new measure at each barline, and
  copied to a new StaffState representing a new line when `new_staff` is called.
  """

  def __init__(self, start_time, clef=None):
    clef = clef or clef_module.TrebleClef()
    self.measure_state = measure.MeasureState(start_time, clef=clef)

  def add_measure(self, start_time):
    """Updates `measure_state` for a new measure.

    Args:
      start_time: The start time of the new measure.

    Copies state which is persisted between measures, and initializes other
    state to the defaults for the new measure.
    """
    self.measure_state = self.measure_state.new_measure(start_time)

  def new_staff(self, start_time):
    """Copies the StaffState to a new staff on a new line.

    Args:
      start_time: Start time of the first measure of the new staff.

    Returns:
      A new StaffState instance.
    """
    # Don't persist the key signature between staves, since we expect it to be
    # at the start of each line.
    return StaffState(start_time, clef=self.measure_state.clef)

  def get_key_signature(self):
    """Returns the key signature at the current point in time."""
    return self.measure_state.key_signature

  def get_time(self):
    """Returns the current time."""
    return self.measure_state.time

  def set_time(self, time):
    """Updates the current time of the current measure.

    Args:
      time: A floating-point time.

    Updates `self.measure_state`.
    """
    self.measure_state.time = time

  def set_clef(self, clef):
    """Updates the clef.

    Args:
      clef: A TrebleClef or BassClef.

    Updates `self.measure_state`.
    """
    self.measure_state.set_clef(clef)
