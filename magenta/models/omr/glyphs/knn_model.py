"""K-Nearest-Neighbors glyph classification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

from magenta.models.omr import musicscore_pb2

from tensorflow.python.estimator.canned import prediction_keys

# k = 3 has the best performance for noteheads, clefs, and sharps. k = 5 seems
# to increase false negatives, so we probably don't want to increase k further
# with our current data.
K_NEAREST_VALUE = 3

NUM_GLYPHS = len(musicscore_pb2.Glyph.Type.keys())


def knn_kmeans_model(centroids, labels, patches=None):
  """The KNN k-means classifier model.

  Args:
    centroids: The k-means centroids NumPy array. Shape `(num_centroids,
      patch_height, patch_width)`.
    labels: The centroid labels NumPy array. Vector with length `num_centroids`.
    patches: Optional input tensor for the patches. If None, a placeholder will
        be used.

  Returns:
    The predictions (class ids) tensor determined from the input patches. Vector
    with the same length as `patches`.
  """
  with tf.name_scope('knn_model'):
    centroids = tf.identity(
        _to_float(tf.constant(_to_uint8(centroids))), name='centroids')
    labels = tf.constant(labels, name='labels')
    centroids_shape = tf.shape(centroids)
    num_centroids = centroids_shape[0]
    patch_height = centroids_shape[1]
    patch_width = centroids_shape[2]
    flattened_centroids = tf.reshape(
        centroids, [num_centroids, patch_height * patch_width],
        name='flattened_centroids')
    if patches is None:
      patches = tf.placeholder(
          tf.float32, (None, centroids.shape[1], centroids.shape[2]),
          name='patches')
    patches_shape = tf.shape(patches)
    flattened_patches = tf.reshape(
        patches, [patches_shape[0], patches_shape[1] * patches_shape[2]],
        name='flattened_patches')
    with tf.name_scope('distance_matrix'):
      distance_matrix = _squared_euclidean_distance_matrix(
          flattened_patches, flattened_centroids)

    # Take the k centroids with the lowest distance to each patch. Wrap the k
    # constant in a tf.identity, which tests can use to feed in another value.
    k_value = tf.identity(tf.constant(K_NEAREST_VALUE), name='k_nearest_value')
    nearest_centroid_inds = tf.nn.top_k(-distance_matrix, k=k_value)[1]
    # Get the label corresponding to each nearby centroids, and reshape the
    # labels back to the original shape.
    nearest_labels = tf.reshape(
        tf.gather(labels, tf.reshape(nearest_centroid_inds, [-1])),
        tf.shape(nearest_centroid_inds),
        name='nearest_labels')
    # Make a histogram of counts for each glyph type in the nearest centroids,
    # for each row (patch).
    length = NUM_GLYPHS
    bins = tf.map_fn(
        lambda row: tf.bincount(row, minlength=length, maxlength=length),
        tf.to_int32(nearest_labels),
        name='bins')
    with tf.name_scope('mode_out_of_k'):
      # Take the argmax of the histogram to get the top prediction. Discard
      # glyph type 1 (NONE) for now.
      mode_out_of_k = tf.argmax(
          bins[:, musicscore_pb2.Glyph.NONE + 1:], axis=1) + 2
      # Force predictions to NONE only if all k nearby centroids were NONE.
      # Otherwise, the non-NONE nearby centroids will contribute to the
      # prediction.
      mode_out_of_k = tf.where(
          tf.equal(bins[:, musicscore_pb2.Glyph.NONE], k_value),
          tf.fill(
              tf.shape(mode_out_of_k), tf.to_int64(musicscore_pb2.Glyph.NONE)),
          mode_out_of_k)
    return tf.identity(mode_out_of_k, name='predictions')


def _to_uint8(values):
  return np.rint(values * 255).astype(np.uint8)


def _to_float(values_t):
  return tf.to_float(values_t) / tf.constant(255.)


def export_knn_model(centroids, labels, export_path):
  """Writes the KNN saved model.

  Args:
    centroids: The k-means centroids NumPy array.
    labels: The labels of the k-means centroids.
    export_path: The output saved model directory.
  """
  g = tf.Graph()
  with g.as_default():
    predictions = knn_kmeans_model(centroids, labels)
    patches = g.get_tensor_by_name('knn_model/patches:0')
    predictions_info = tf.saved_model.utils.build_tensor_info(predictions)
    patches_info = tf.saved_model.utils.build_tensor_info(patches)
    with tf.Session(graph=g) as sess:
      builder = tf.saved_model.builder.SavedModelBuilder(export_path)
      builder.add_meta_graph_and_variables(
          sess, ['serve'],
          signature_def_map={
              tf.saved_model.signature_constants.
              DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  tf.saved_model.signature_def_utils.build_signature_def(
                      inputs={'input': patches_info},
                      outputs={
                          prediction_keys.PredictionKeys.CLASS_IDS:
                              predictions_info
                      }),
          })
      builder.save()


def _squared_euclidean_distance_matrix(a, b):
  # Trick for computing the squared Euclidean distance matrix.
  # Entry (i, j) = a[i].sum() + b[j].sum() - 2 * (a[i] * b[j]).sum()
  #              = sum_k (a[i, k] + b[j, k] - 2 * a[i, k] * b[j, k])
  #              = sum_k (a[i, k] - b[j, k]) ** 2
  a_sum = tf.reshape(tf.reduce_sum(a, axis=1), [-1, 1])  # column vector
  b_sum = tf.reshape(tf.reduce_sum(b, axis=1), [1, -1])  # row vector

  return a_sum + b_sum - 2 * tf.matmul(a, b, transpose_b=True)
