"""Hyperparameter management."""
import ast
import tensorflow as tf
import yaml

from magenta.models.wayback.lib.namespace import Namespace as NS


schema = NS(
    (name, NS(name=name, description=description, default=default))
    for name, (description, default) in dict(
        sampling_frequency=("desired waveform time resolution in Hz", 44100),
        bit_depth=("desired waveform amplitude resolution in bits", 8),
        data_dim=("data dimensionality (usually inferred)", 256),

        initial_learning_rate=("initial learning rate", 0.002),
        decay_patience=("how long to wait for improvement before decaying the"
                        " learning rate", 100),
        decay_rate=("rate of decay of learning rate", 0.1),
        clip_norm=("ratio for gradient clipping_by_norm", 1),

        batch_size=("number of examples in minibatch", 100),
        use_bn=("whether to use batch normalizatin", False),
        activation=("recurrent activation function to use (tanh/elu/identity)",
                    "tanh"),
        io_sizes=("layer sizes for input and output MLPs", [512]),
        weight_decay=("L2 weight decay coefficient", 1e-7),

        segment_length=("length of truncated backpropagation", 1000),
        chunk_size=("number of samples per model step", 1),
        layout=("recurrent connection pattern (stack/wayback)", "stack"),
        cell=("recurrent cell (rnn/lstm/gru)", "lstm"),
        periods=("update interval for each layer, from bottom to top. only"
                 " used for the wayback layout", [1000]),
        layer_sizes=("number of hidden units in each layer, from bottom to"
                     " top.", [1000]),
        vskip=("vertical skip connections between all layers", False),

        unroll_layer_count=("number of upper layers to move outside the while"
                            " loop. only used for the wayback layout", 0),
        carry=("whether to carry state between cycles or restart based on"
               " context. only used for the wayback layout", True)
    ).items())


def get_defaults(**overrides):
  """Get default hyperparameters.

  Args:
    **overrides: overrides for a subset of hyperparameters.

  Raises:
    ValueError: If an override refers to a nonexistent hyperparameter or the
                specified value is of a different type than the default value.

  Returns:
    A Namespace with (possibly overridden) defaults.
  """
  hp = NS((name, hyperparameter.default)
          for name, hyperparameter in schema.Items())
  for name, value in overrides.items():
    if name not in hp:
      raise ValueError("value provided for nonexistent hyperparameter %s"
                       % name)
    # TODO(cotim): deep typecheck
    if type(value) is not type(hp[name]):
      raise ValueError("value %s (%s) provided for hyperparameter %s does not"
                       " match type of default %s (%s)"
                       % (value, type(value), name,
                          hp[name], type(hp[name])))
    hp[name] = value
  return hp


def parse(string):
  """Parse a hyperparameter string, filling in defaults.

  A hyperparameter string consists of a dictionary literal. To make command-line
  life easier, barewords are converted to strings.

  Example:
    hp = parse("{a: 1, b: [2, 3, four], c: {d: five, e: False}}")
    assert hp == NS(a=1, b=[2, 3, "four"], c=NS(d="five", e=False))

  Args:
    string: A hyperparameter string.

  Returns:
    A Namespace containing the designated hyperparameters.
  """
  return get_defaults(**parse_value(string).AsDict())


class ParseError(Exception):
  pass


def parse_value(expr):
  """Parse a hyperparameter value.

  A value can be any Python literal. Barewords are converted to strings.
  Dictionaries are converted to Namespaces.

  Args:
    expr: value expression as a string or `ast.expr`

  Raises:
    ParseError: if `expr` is not a literal expression.

  Returns:
    The value represented by `expr`.
  """
  if isinstance(expr, basestring):
    expr = ast.parse(expr).body[0].value

  if isinstance(expr, ast.Num):
    return expr.n
  elif isinstance(expr, ast.Str):
    return expr.s
  elif isinstance(expr, ast.Name):
    try:
      # True/False are represented as Names -_-
      return ast.literal_eval(expr.id)
    except ValueError:
      # interpret as string
      return expr.id
  elif isinstance(expr, ast.List):
    return list(map(parse_value, expr.elts))
  elif isinstance(expr, ast.Tuple):
    return tuple(map(parse_value, expr.elts))
  elif isinstance(expr, ast.Dict):
    return NS((parse_key(key), parse_value(value))
              for key, value in zip(expr.keys, expr.values))
  else:
    raise ParseError("invalid value", expr)


def parse_key(expr):
  """Parse a hyperparameter key.

  A key can be a string or a bareword.

  Args:
    expr: key expression as a string or `ast.expr`

  Raises:
    ParseError: if `expr` is not a string or bareword expression.

  Returns:
    The key represented by `expr`.
  """
  if isinstance(expr, basestring):
    expr = ast.parse(expr).body[0].value

  if isinstance(expr, ast.Str):
    return expr.s
  elif isinstance(expr, ast.Name):
    return expr.id
  else:
    raise ParseError("invalid key", expr)


def dump(filelike, hp):
  """Dump hyperparameters to a YAML file.

  Args:
    filelike: a file object or a path to a file.
    hp: Namespace object to dump.
  """
  if not isinstance(filelike, file):
    filelike = tf.gfile.Open(filelike, "w")
  filelike.write(yaml.dump(hp.AsDict()))


def load(filelike):
  """Load hyperparameters from a file.

  Args:
    filelike: a file object or a path to a file.

  Returns:
    Namespace containing hyperparameters.
  """
  if not isinstance(filelike, file):
    filelike = tf.gfile.Open(filelike)
  hp = yaml.load(filelike.read())
  hp = get_defaults(**hp)
  return hp
