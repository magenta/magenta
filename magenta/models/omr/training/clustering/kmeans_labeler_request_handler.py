"""The k-means labeler HTTP request handler.

Displays the clusters on an HTML page with dropdowns for each cluster's label.
On POST, updates the output to include the cluster labels.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import cgi

# internal imports
from mako import template as mako_template
import numpy as np
from PIL import Image
import six
from six import moves
from six.moves import BaseHTTPServer
from six.moves import http_client

from magenta.models.omr.protobuf import musicscore_pb2

from tensorflow.core.example import example_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import resource_loader


class LabelerHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
  """The HTTP request handler for the k-means labeler server.

  Attributes:
    clusters: NumPy array of clusters. Shape
        (num_clusters, patch_height, patch_width).
    output_path: Path to write the TFRecords.
  """

  def __init__(self, clusters, output_path, *args):
    self.clusters = clusters
    self.output_path = output_path
    BaseHTTPServer.BaseHTTPRequestHandler.__init__(self, *args)

  def do_GET(self):
    template_path = resource_loader.get_path_to_datafile(
        'kmeans_labeler_template.html')
    template = mako_template.Template(open(template_path).read())
    page = create_page(self.clusters, template)
    self.send_response(http_client.OK)
    self.send_header('Content-Type', 'text/html; charset=utf-8')
    self.end_headers()
    self.wfile.write(page)

  def do_POST(self):
    post_vars = cgi.parse_qs(
        self.rfile.read(int(self.headers.getheader('content-length'))))
    labels = [
        post_vars['cluster%d' % i][0]
        for i in moves.range(self.clusters.shape[0])
    ]
    examples = create_examples(self.clusters, labels)

    with tf_record.TFRecordWriter(self.output_path) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
    self.send_response(http_client.OK)
    self.end_headers()
    self.wfile.write('Success')  # printed in the labeler alert


def create_page(clusters, template):
  """Renders the labeler HTML.

  Args:
    clusters: NumPy array of clusters.
    template: Mako template for the page.

  Returns:
    The labeler HTML string.
  """

  # Tuples (index, preview, is_content).
  cluster_info = [
      (i,) + _process_cluster(cluster) for i, cluster in enumerate(clusters)
  ]
  content_clusters = [cluster for cluster in cluster_info if cluster[2]]
  non_content_clusters = [cluster for cluster in cluster_info if not cluster[2]]
  return template.render(
      content_clusters=content_clusters,
      empty_clusters=non_content_clusters,
      # Skip the unknown glyph type.
      glyph_types=musicscore_pb2.Glyph.Type.keys()[1:])


def _process_cluster(cluster):
  """Processes a cluster cluster image.

  Args:
    cluster: 2D NumPy array.

  Returns:
    The preview image (PNG encoded in an HTML data URL).
    is_content: boolean; whether the cluster is considered content.
  """
  image_arr = create_highlighted_image(cluster)
  image = Image.fromarray(image_arr)
  buf = six.BytesIO()
  image.save(buf, 'PNG', optimize=True)
  buf.seek(0)
  preview = b'data:image/png;base64,' + base64.b64encode(buf.read())
  # The cluster is likely non-content (does not contain a glyph) if the
  # max standard deviation across all rows or all columns is low. Show those
  # patches at the bottom of the page so that they can still be double checked
  # by hand.
  is_content = min(cluster.std(axis=0).max(), cluster.std(axis=1).max()) > 0.1
  return preview, is_content


def create_highlighted_image(cluster, enlarge_ratio=10):
  """Enlarges the "cluster" image and draws a crosshairs highlight.

  Args:
    cluster: 2D NumPy array.
    enlarge_ratio: Scale of the output image.

  Returns:
    An enlarged and highlighted image as a NumPy array. 3D (height, width, 3).
        A "crosshairs" pattern is drawn which highlights the center pixel of the
        image.
  """
  # Enlarge the image, and repeat on the 3rd axis to get RGB channels.
  image_arr = np.repeat(
      np.repeat(
          np.repeat((cluster * 255).astype(np.uint8)[:, :, None], enlarge_ratio,
                    axis=0),
          enlarge_ratio,
          axis=1),
      3,
      axis=2)
  # Calculate the vertical and horizontal slice of the image to be highlighted.
  vertical_start = (image_arr.shape[0] - enlarge_ratio) // 2
  vertical_stop = (image_arr.shape[0] + enlarge_ratio) // 2
  vertical_slice = slice(vertical_start, vertical_stop)
  horiz_slice = slice(image_arr.shape[1] // 2 - 5, image_arr.shape[1] // 2 + 6)
  # Highlight the vertical slice of the image to be partially red.
  image_arr[vertical_slice] = (
      image_arr[vertical_slice] / 2 + np.array([127., 0., 0.]))
  # Highlight the horizontal slice of the image, avoiding the section that was
  # already highlighted.
  image_arr[:vertical_start, horiz_slice] = (
      image_arr[:vertical_start, horiz_slice] / 2 + np.array([127., 0., 0.]))
  image_arr[vertical_stop:, horiz_slice] = (
      image_arr[vertical_stop:, horiz_slice] / 2 + np.array([127., 0., 0.]))
  return image_arr


def create_examples(clusters, labels):
  """Creates Examples from the clusters and label strings.

  Args:
    clusters: NumPy array of shape (num_clusters, patch_height, patch_width).
    labels: List of string labels, which are names in musicscore_pb2.Glyph.Type.
        Length `num_clusters`.

  Returns:
    A list of Example protos of length `num_clusters`.
  """
  examples = []
  for cluster, label in zip(clusters, labels):
    example = example_pb2.Example()
    features = example.features
    features.feature['patch'].float_list.value.extend(cluster.ravel())
    features.feature['height'].int64_list.value.append(cluster.shape[0])
    features.feature['width'].int64_list.value.append(cluster.shape[1])
    label_num = musicscore_pb2.Glyph.Type.Value(label)
    example.features.feature['label'].int64_list.value.append(label_num)
    examples.append(example)
  return examples
