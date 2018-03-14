"""Tests for the k-means labeler request handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

# internal imports
from absl.testing import absltest
from mako import template as mako_template
import numpy as np

from magenta.models.omr.protobuf import musicscore_pb2
from magenta.models.omr.training.clustering import kmeans_labeler_request_handler

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.platform import resource_loader


class KmeansLabelerRequestHandlerTest(absltest.TestCase):

  def testCreatePage(self):
    num_clusters = 10
    clusters = np.random.random((num_clusters, 12, 14))
    template_src = resource_loader.get_path_to_datafile(
        'kmeans_labeler_template.html')
    template = mako_template.Template(open(template_src).read())
    html = kmeans_labeler_request_handler.create_page(clusters, template)
    self.assertEqual(len(re.findall('<img', html)), num_clusters)
    self.assertEqual(len(re.findall('<select', html)), num_clusters)

  def testCreateExamples(self):
    clusters = np.random.random((5, 12, 14))
    labels = ['NONE', 'CLEF_TREBLE', 'NONE', 'NOTEHEAD_FILLED', 'CLEF_BASS']
    examples = kmeans_labeler_request_handler.create_examples(clusters, labels)
    self.assertEqual(len(examples), 5)
    for i in range(5):
      glyph_type = musicscore_pb2.Glyph.Type.Value(labels[i])
      features = feature_pb2.Features()
      features.feature['patch'].float_list.value.extend(clusters[i].ravel())
      features.feature['height'].int64_list.value.append(12)
      features.feature['width'].int64_list.value.append(14)
      features.feature['label'].int64_list.value.append(glyph_type)
      self.assertEqual(examples[i], example_pb2.Example(features=features))


if __name__ == '__main__':
  absltest.main()
