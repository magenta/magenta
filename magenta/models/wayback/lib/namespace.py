"""Utilities for dealing with collections of variables."""
from collections import OrderedDict as odict
import itertools as it


class Namespace(object):
  """A class for key-value pairs with less noisy syntax than dicts.

  Namespace is like an OrderedDict that exposes its members as attributes. Keys
  are constrained to be strings matching [a-z_]+. Capitalized names are reserved
  for instance methods.

  Namespace is designed to be used hierarchically; values can be Namespaces or
  lists or general opaque objects. The `Flatten` method makes it
  straightforward to flatten such a structure into a list for those terrible
  interfaces that require you to do that, and `UnflattenLike` allows you to
  restore the structure on the other end.
  """

  def __init__(self, *args, **kwargs):
    # pylint: disable=invalid-name
    self.Data = odict(*args, **kwargs)

  def __getitem__(self, *args, **kwargs):
    return self.Data.__getitem__(*args, **kwargs)

  def __setitem__(self, *args, **kwargs):
    return self.Data.__setitem__(*args, **kwargs)

  def __contains__(self, *args, **kwargs):
    return self.Data.__contains__(*args, **kwargs)

  def __getattr__(self, key):
    if key[0].isupper():
      return self.__dict__[key]
    return self.Data[key]

  def __setattr__(self, key, value):
    if key[0].isupper():
      self.__dict__[key] = value
    self.Data[key] = value

  def __iter__(self):
    return self.Keys()

  def Keys(self):
    for key in self.Data:
      if not key[0].isupper():
        yield key

  def Values(self):
    for key in self.Keys():
      yield self[key]

  def Items(self):
    for key in self.Keys():
      yield (key, self[key])

  def __eq__(self, other):
    if not isinstance(other, Namespace):
      return False
    if self is other:
      return True
    for key in self:
      if key not in other or self[key] != other[key]:
        return False
    for key in other:
      if key not in self or other[key] != self[key]:
        return False
    return True

  def __ne__(self, other):
    return not self == other

  def AsDict(self):
    """Convert to an OrderedDict."""
    other = odict()
    for key in self:
      other[key] = self[key]
    return other

  def __repr__(self):
    return "Namespace(%s)" % ", ".join("%s=%s" % (key, repr(value))
                                       for key, value in self.Items())

  def __str__(self):
    return "{%s}" % ", ".join("%s: %s" % (key, str(value))
                              for key, value in self.Items())

  def Update(self, other):
    """Update `self` with key-value pairs from `other`.

    Args:
      other: Namespace or dict-like
    """
    if isinstance(other, Namespace):
      # python has a stinky manner of detecting dict-likes
      self.Data.update(other.AsDict())
    else:
      self.Data.update(other)

  def Extract(self, *keyss):
    """Extract a subnamespace from `self` with the given keys.

    Each of `keyss` is expected to be a string containing space-separated key
    expressions. A key expression can be either a single key or a chain of keys
    for direct access to nested Namespaces. E.g.:

        Namespace(v=2, w=Namespace(x=1, y=Namespace(z=0)))._extract("w.y v")

    would return

        Namespace(w=Namespace(y=Namespace(z=0)), v=2)

    The returned object is a Namespace with the same structure as `self`, except
    that each Namespace contained to it is narrowed to the selected keys.

    Args:
      *keyss: A sequence of strings each containing space-separated key
              expressions.

    Returns:
      The narrowed-down namespace.
    """
    other = Namespace()
    for keys in keyss:
      for key in keys.split():
        trail = key.split(".")
        if len(trail) == 1:
          other[trail[0]] = self[trail[0]]
        else:
          # get fancy: extract members from nested namespaces
          if trail[0] not in other:
            other[trail[0]] = Namespace()
          other[trail[0]].Update(self[trail[0]].Extract(".".join(trail[1:])))
    return other

  def Get(self, path, default=None):
    """Get an object from a Namespace tree.

    This is analogous to the `get` method on dicts, which does not raise
    KeyError if the key is not in the dict but rather returns a default value
    provided by the caller. This method performs a chain of lookups in a
    Namespace tree, returning the provided default if any of the lookups
    triggers a KeyError. For example:

      ns = Namespace(x=Namespace(y=1, z=Namespace()))
      assert ns.x.y == 1
      assert ns.Get("x.y") == 1
      assert ns.Get("x.z.w", None) == None

    Args:
      path: the path to the object.
      default: the value to return if the path does not exist.

    Returns:
      The object or `default`.
    """
    obj = self
    for key in path.split("."):
      try:
        obj = obj.Data[key]
      except KeyError:
        return default
    return obj

  @staticmethod
  def Flatten(x):
    """Flatten a (possibly nested) Namespace into a list.

    Constructs a list by traversing the tree `x` of nested (lists of) Namespaces
    in order. Descends into Namespaces, tuples and lists. For example:

      ns = Namespace()
      ns.x = [1, 2, Namespace(y=3)]
      ns.z = Namespace(w=[4, 5, (6, 7)])
      assert Namespace.Flatten(ns) == [1, 2, 3, 4, 5, 6, 7]
      assert Namespace.Flatten(ns.x) == [1, 2, 3]
      assert Namespace.Flatten(ns.z.w) == [4, 5, 6, 7]

    Args:
      x: A Namespace, tuple, list or opaque object to flatten.

    Returns:
      The leaf nodes of `x` collected into a list.
    """
    if isinstance(x, (tuple, list)):
      return list(it.chain.from_iterable(map(Namespace.Flatten, x)))
    elif isinstance(x, Namespace):
      return list(it.chain.from_iterable(Namespace.Flatten(x[key])
                                         for key in x))
    else:
      return [x]

  @staticmethod
  def UnflattenLike(xform, yflat):
    """Unflatten a list into a (possibly nested) Namespace.

    This is the inverse of `Flatten`. The structure information is taken from
    the model `xform`. E.g.:

        xflat = Flatten(xform)
        yflat = function_that_likes_lists(xflat)
        yform = UnflattenLike(xform, yflat)

    Args:
      xform: A Namespace tree according to which to structure yflat.
      yflat: A flat list of opaque objects.

    Returns:
      A Namespace tree structured like `xform` with values from `yflat`.
    """
    def _UnflattenLike(xform, yflat):  # pylint: disable=missing-docstring
      if isinstance(xform, (tuple, list)):
        yform = []
        for xelt in xform:
          yelt, yflat = _UnflattenLike(xelt, yflat)
          yform.append(yelt)
      elif isinstance(xform, Namespace):
        yform = Namespace()
        for key in xform:
          yelt, yflat = _UnflattenLike(xform[key], yflat)
          yform[key] = yelt
      else:
        yform, yflat = yflat[0], yflat[1:]
      return yform, yflat
    yform, yflat_leftover = _UnflattenLike(xform, yflat)
    assert not yflat_leftover
    return yform

  @staticmethod
  def Copy(x):
    """Copy a Namespace tree.

    Performs a shallow clone, in the sense that the structure (`Namespace`s,
    lists, tuples) is recreated but the leaf nodes (everything else) are
    copied by reference.

    Args:
      x: The Namespace tree to clone.

    Returns:
      A Namespace tree isomorphic to `x` and with leaf nodes identical to those
      in `x`, but with independent structure.
    """
    return Namespace.UnflattenLike(x, Namespace.Flatten(x))

  @staticmethod
  def FlatCall(fn, tree):
    """Call a list-minded function with objects from a Namespace tree.

    This is a wrapper around `Flatten` and `UnflattenLike` for the common
    case of calling a function that takes and returns lists. The tree is
    flattened into a list, which is passed as an argument to `fn`. `fn`
    returns a list of corresponding outputs, which is unflattened into the
    same structure as `tree` and subsequently returned.

    Args:
      fn: The offending function.
      tree: The Namespace tree to flatten and unflatten.

    Returns:
      The values returned from `fn`, with structure corresponding to `tree`.
    """
    return Namespace.UnflattenLike(tree, fn(Namespace.Flatten(tree)))

  @staticmethod
  def FlatZip(trees, path=None):
    """Zip values from multiple Namespace trees.

    Narrows each of the `trees` to `path` if given, then flattens each tree
    and zips it up.  Example:

        mapping = dict(FlatZip([keys, values]))

    Args:
      trees: the Namespace trees to zip.
      path: the path to which to narrow the trees.

    Raises:
      ValueError: if the `trees` are not isomorphic and `path` is not given.

    Returns:
      An iterator over tuples with corresponding elements from `trees`.
    """
    trees = list(trees)
    if path:
      trees = [Namespace.Extract(tree, path) for tree in trees]
    if not Namespace.Isomorphic(*trees):
      raise ValueError("Trees not isomorphic")
    return zip(*list(map(Namespace.Flatten, trees)))

  @staticmethod
  def Isomorphic(*trees):
    """Test whether Namespace trees are mutually isomorphic.

    Two trees are isomorphic iff they have the same structure:
      * sequences a and b are isomorphic iff they have the same length and
        all parallel elements a[i] and b[i] are isomorphic.
      * Namespaces a and b are isomorphic iff they have the same set of keys
        and all parallel elements a[key] and b[key] are isomorphic.
      * leaf nodes a and b are isomorphic.

    Args:
      *trees: the trees to compare.

    Returns:
      True if the trees are isomorphic, False otherwise.
    """
    deepkey_lists = [sorted(Namespace.DeepKeys(tree)) for tree in trees]
    if any(len(deepkey_list) != len(deepkey_lists[0])
           for deepkey_list in deepkey_lists):
      return False
    for parallel_deepkeys in zip(*deepkey_lists):
      if any(deepkey != parallel_deepkeys[0] for deepkey in parallel_deepkeys):
        return False
    return True

  @staticmethod
  def DeepKeys(tree):
    """Get the keys to all leaf nodes in a Namespace tree.

    Args:
      tree: the tree to traverse.

    Yields:
      keys to each node in the tree, in order, represented by tuples of
           shallow keys.
    """
    if isinstance(tree, (tuple, list)):
      for i, subtree in enumerate(tree):
        for subkey in Namespace.DeepKeys(subtree):
          yield (i,) + subkey
    elif isinstance(tree, Namespace):
      for key, subtree in tree.Items():
        for subkey in Namespace.DeepKeys(subtree):
          yield (key,) + subkey
    else:
      yield ()
