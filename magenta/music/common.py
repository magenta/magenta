"""Common functions for magenta.music."""

import collections

def make_callback_dict(callback_dict):
  new_dict = collections.defaultdict(lambda: lambda *_: None)
  if callback_dict is not None:
    new_dict.update(callback_dict)
  return new_dict
