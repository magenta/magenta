"""Simple memoizer for a function/method/property."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MemoizedFunction(object):
  """Decorates a function to be memoized.

  Caches all invocations of the function with unique arguments. The arguments
  must be hashable.

  Decorated functions are not threadsafe. This decorator is currently used for
  TensorFlow graph construction, which happens in a single thread.
  """

  def __init__(self, function):
    self._function = function
    self._results = {}

  def __call__(self, *args):
    """Calls the function to be memoized.

    Args:
      *args: The args to pass through to the function. Keyword arguments are not
          supported.

    Raises:
      TypeError if an argument is unhashable.

    Returns:
      The memoized return value of the wrapped function. The return value will
      be computed exactly once for each unique argument tuple.
    """
    if args in self._results:
      return self._results[args]
    self._results[args] = self._function(*args)
    return self._results[args]
