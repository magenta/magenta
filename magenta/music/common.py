"""Common functions for `magenta.music`."""

import collections


def make_callback_dict(callback_dict=None):
  """Create a `defaultdict` of callback functions.

  Args:
    callback_dict: A dict with callback functions as values, or None.
  Returns:
    A `collections.defaultdict` instance with a dummy function as the default
    value and with the same contents as `callback_dict`.
  """
  new_dict = collections.defaultdict(lambda: _dummy_fn)
  if callback_dict is not None:
    new_dict.update(callback_dict)
  return new_dict


def _dummy_fn(*unused_args, **unused_kwargs):
  pass
