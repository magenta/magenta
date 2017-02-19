"""Tensorflow utility functions."""
import math
import tensorflow as tf

from magenta.models.wayback.lib.namespace import Namespace as NS


def shaped_one_hot(indices, shape, **one_hot_kwargs):
  """Like `tf.one_hot`, but set shape information.

  Args:
    indices: see `tf.one_hot`.
    shape: constant shape sequence used to infer the `depth` argument to
           `tf.one_hot` and used to set shape using `set_shape`.
    **one_hot_kwargs: passed onto `tf.one_hot`.

  Returns:
    The same thing as `tf.one_hot`, but with shape information.
  """
  vectors = tf.one_hot(indices, depth=shape[-1], **one_hot_kwargs)
  vectors.set_shape(shape)
  return vectors


def meanlogabs(x, reduction_indices=None):
  """Compute the average log-magnitude of values in a `Tensor`.

  Args:
    x: a `Tensor`.
    reduction_indices: passed on to `tf.reduce_mean`.

  Returns:
    mean(log(1 + abs(x)))
  """
  return tf.reduce_mean(tf.log(1 + tf.abs(x)),
                        reduction_indices=reduction_indices)


def layers(xs, sizes, scope=None, **layer_kwargs):
  """Sequence of nonlinear layers.

  Args:
    xs: input matrices of shape [batch, ?].
    sizes: number of units in each layer.
    scope: containing scope name for Tensorflow variables.
    **layer_kwargs: passed onto `layer`.

  Returns:
    The output of the last layer.
  """
  xs = list(xs)
  with tf.variable_op_scope(xs, scope or "layers"):
    for i, size in enumerate(sizes):
      with tf.variable_op_scope(xs, str(i)):
        xs = [layer(xs, output_dim=size, **layer_kwargs)]
    return xs[0]


def layer(xs, fn=tf.nn.elu, use_bn=True, use_bias=True, **project_terms_kwargs):
  """Linear projection followed by an activation function.

  Args:
    xs: the terms to project.
    fn: the activation function.
    use_bn: whether to apply batch normalization.
    use_bias: whether to include a bias term.
    **project_terms_kwargs: passed onto `project_terms`.

  Returns:
    The output of the layer.
  """
  project_terms_kwargs.setdefault("use_bn", use_bn)
  project_terms_kwargs.setdefault("use_bias", use_bias)
  return fn(project_terms(xs, **project_terms_kwargs))


def project_terms(xs, output_dim=None, use_bn=False, use_bias=True, scope=None):
  """Linearly project multiple terms and sum their projections.

  Args:
    xs: the terms to project.
    output_dim: the dimensionality of the projection.
    use_bn: whether to batch-normalize each of the terms before adding them up.
    use_bias: whether to add a bias.
    scope: containing scope name for Tensorflow variables.

  Returns:
    The sum of the projections of `xs`.
  """
  with tf.variable_op_scope([xs], scope or "project_terms"):
    if use_bn:
      # batch-normalize each projection separately before summing
      projected_xs = [
          project(x, output_dim=output_dim, use_bias=False, scope=str(i))
          for i, x in enumerate(xs)]
      normalized_xs = [
          batch_normalize(x, beta=0, scope=str(i))
          for i, x in enumerate(projected_xs)]
      y = sum(normalized_xs)
    else:
      # concatenate terms and do one big projection
      y = project(tf.concat(1, xs), output_dim=output_dim, use_bias=False)
    if use_bias:
      y += tf.get_variable("beta", shape=[output_dim],
                           initializer=tf.constant_initializer(0))
    return y


def project(x, output_dim=None, use_bias=True, scope=None):
  """Linearly project `x`.

  Args:
    x: the quantity to project.
    output_dim: the dimensionality of the projection.
    use_bias: whether to add a bias.
    scope: containing scope name for TensorFlow variables.

  Returns:
    The projection of `x`.
  """
  with tf.variable_op_scope([x], scope or "project"):
    input_dim = x.shape[-1]
    w = tf.get_variable("w", shape=[input_dim, output_dim],
                        initializer=tf.truncated_normal_initializer(
                            stddev=math.sqrt(1. / output_dim)),
                        regularizer=lambda w: w)
    y = tf.matmul(x, w)
    if use_bias:
      y += tf.get_variable("beta", shape=[output_dim],
                           initializer=tf.constant_initializer(0))
    return y


def batch_normalize(x, beta=None, gamma=None, epsilon=1e-5, scope=None):
  """Batch-normalize `x`.

  Args:
    x: the quantity to batch-normalize.
    beta: Variable representing the desired mean of the output.
          If None, a Variable will be created.
    gamma: Variable representing the desired standard deviation of the output.
           If None, a Variable will be created.
    epsilon: Variance estimate regularizer.
    scope: containing scope name for Tensorflow variables.

  Returns:
    `x`, batch-normalized.
  """
  # TODO(cotim): implement population statistics
  with tf.variable_op_scope([x, beta, gamma], scope or "bn"):
    mean, variance = tf.nn.moments(x, [0])
    if gamma is None:
      gamma = tf.get_variable("gamma", shape=x.shape[-1],
                              initializer=tf.constant_initializer(0.1))
    if beta is None:
      beta = tf.get_variable("beta", shape=x.shape[-1],
                             initializer=tf.constant_initializer(0))
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


# pylint: disable=redefined-outer-name
def while_loop(cond, body, loop_vars, **kwargs):
  """Like `tf.while_loop` but with structured `loop_vars`.

  Args:
    cond: as in `tf.while_loop`, but takes a single `loop_vars` argument.
    body: as in `tf.while_loop`, but takes and returns a single `loop_vars`
          tree which it is allowed to modify.
    loop_vars: as in `tf.while_loop`, but consists of a Namespace tree.
    **kwargs: passed onto `tf.while_loop`.

  Returns:
    A Namespace tree structure containing the final values of the loop
    variables.
  """
  def _cond(*flat_vars):
    return cond(NS.UnflattenLike(loop_vars, flat_vars))

  def _body(*flat_vars):
    return NS.Flatten(body(NS.UnflattenLike(loop_vars, flat_vars)))

  return NS.UnflattenLike(
      loop_vars,
      tf.while_loop(cond=_cond, body=_body,
                    loop_vars=NS.Flatten(loop_vars),
                    **kwargs))
# pylint: enable=redefined-outer-name


def cond(pred, fn1, fn2, prototype, **kwargs):
  """Like `tf.cond` but with structured collections of variables.

  Args:
    pred: boolean Tensor, as in `tf.cond`.
    fn1: a callable representing the `then` branch as in `tf.cond`, but
         may return an arbitrary Namespace tree.
    fn2: a callable representing the `else` branch as in `tf.cond`, but
         may return an arbitrary Namespace tree.
    prototype: an example Namespace tree to indicate the structure of the
               values returned from `fn1` and `fn2`.
    **kwargs: passed onto `tf.cond`.

  Returns:
    Like `tf.cond`, except structured like `prototype`.
  """
  def wrap_branch(fn):
    def wrapped_branch():
      tree = fn()
      liszt = NS.Flatten(tree)
      return liszt
    return wrapped_branch

  results = tf.cond(pred, wrap_branch(fn1), wrap_branch(fn2), **kwargs)
  # tf.cond unpacks singleton lists returned from fn1, fn2 -_-
  if not isinstance(results, (tuple, list)):
    results = [results]
  # need a prototype to unflatten because at this point neither fn1 nor fn2
  # have been called
  tree3 = NS.UnflattenLike(prototype, results)
  return tree3


def leaky_relu(x, alpha=1e-2):
  """Compute the Leaky ReLU activation function.

  Args:
    x: preactivation
    alpha: slope at x < 0

  Returns:
    The activation value.
  """
  return tf.maximum(alpha * x, x)


def sample(logits, temperature=1.0):
  """Sample from softmax distribution over targets.

  Args:
    logits: Possibly unnormalized softmax energies to sample from.
    temperature: Softmax temperature; higher is more random.

  Returns:
    Sample, as an integer index.
  """
  if temperature != 1:
    logits /= temperature
  index = tf.to_int32(tf.multinomial(logits, 1)[:, 0])
  return index
