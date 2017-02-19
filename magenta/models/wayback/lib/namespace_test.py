from collections import OrderedDict
import unittest

from magenta.models.wayback.lib.namespace import Namespace as NS


class NamespaceTest(unittest.TestCase):

  def setUp(self):
    pass

  def testMisc(self):
    ns = NS()
    ns.w = 0
    ns["x"] = 3
    ns.x = 1
    ns.y = NS(z=2)
    self.assertEqual(list(ns), ["w", "x", "y"])
    self.assertEqual(list(ns.Keys()), ["w", "x", "y"])
    self.assertEqual(list(ns.Values()), [0, 1, NS(z=2)])
    self.assertEqual(list(ns.Items()),
                     [("w", 0), ("x", 1), ("y", NS(z=2))])
    self.assertEqual(
        ns.AsDict(),
        OrderedDict([("w", 0), ("x", 1), ("y", NS(z=2))]))
    ns.Update(ns.y)
    self.assertEqual(list(ns), ["w", "x", "y", "z"])
    self.assertEqual(list(ns.Keys()), ["w", "x", "y", "z"])
    self.assertEqual(list(ns.Values()), [0, 1, NS(z=2), 2])
    self.assertEqual(list(ns.Items()),
                     [("w", 0), ("x", 1), ("y", NS(z=2)), ("z", 2)])
    self.assertEqual(
        ns.AsDict(),
        OrderedDict([("w", 0), ("x", 1), ("y", NS(z=2)), ("z", 2)]))

  def testExtract(self):
    ns = NS(v=2, w=NS(x=1, y=NS(z=0))).Extract("w.y v")
    self.assertEqual(ns.v, 2)
    self.assertEqual(ns.w, NS(y=NS(z=0)))
    self.assertEqual(ns.w.y, NS(z=0))

  def testGet(self):
    ns = NS(foo=NS(bar="baz"))
    self.assertRaises(KeyError, lambda: ns["foo"]["baz"])
    self.assertIsNone(ns.Get("foo.baz"))
    x = object()
    self.assertEqual(x, ns.Get("foo.baz", x))
    self.assertEqual("baz", ns.Get("foo.bar"))
    self.assertEqual(NS(bar="baz"), ns.Get("foo"))

  def testFlattenUnflatten(self):
    before = NS(v=2, w=NS(x=1, y=NS(z=0)))
    flat = NS.Flatten(before)
    after = NS.UnflattenLike(before, flat)
    self.assertEqual(before, after)

  def testCopy(self):
    before = NS(v=2, w=NS(x=1, y=NS(z=0)))
    after = NS.Copy(before)
    self.assertEqual(before, after)
    self.assertTrue(all(a is b for a, b in
                        zip(NS.Flatten(after), NS.Flatten(before))))

  def testFlatCallFlatZip(self):
    before = NS(v=2, w=NS(x=1, y=NS(z=0)))
    after = NS.FlatCall(lambda xs: [2 * x for x in xs], before)
    self.assertEqual(NS(v=4, w=NS(x=2, y=NS(z=0))), after)
    self.assertItemsEqual([(2, 4), (1, 2), (0, 0)],
                          list(NS.FlatZip([before, after])))
    after.w.y.a = 6
    self.assertRaises(ValueError, lambda: NS.FlatZip([before, after]))
    self.assertItemsEqual([(2, 4), (0, 0)],
                          list(NS.FlatZip([before, after], "v w.y.z")))

  def testDeepKeys(self):
    ns = NS(v=2, w=NS(x=1, y=[3, NS(z=0)]))
    self.assertItemsEqual([("v",),
                           ("w", "x"),
                           ("w", "y", 0),
                           ("w", "y", 1, "z")],
                          list(NS.DeepKeys(ns)))

if __name__ == "__main__":
  unittest.main()
