# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for polyphonic RNN."""

from collections import OrderedDict
import copy
import os
import uuid

# internal imports

import numpy as np
from scipy import linalg
import tensorflow as tf

import magenta.music as mm
from magenta.protobuf import music_pb2

# This list represents the duration times (in seconds) that are supported.
# If an input NoteSequence contains a note duration that is not in this list,
# the entire NoteSequence will be discarded.
# TODO(fjord): this filtering should happen at dataset creation time.
TIME_CLASSES = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

##
# begin metautils
##


def shape(x):
  """Returns tensor shape as a tuple of integers, None dimensions as -1."""
  return tuple([-1 if d is None else d for d in x.get_shape().as_list()])


def ndim(x):
  """Returns the number of dimensions of the TensorFlow variable `x`.

  Args:
    x: A TensorFlow tensor object.

  Returns:
    An integer denoting the number of dimensions in `x`.
  """
  return x.get_shape().ndims


def dot(a, b):
  # Generalized dot for nd sequences, assumes last axis is projection
  # b must be rank 2
  a_tup = shape(a)
  b_tup = shape(b)
  a_i = tf.reshape(a, [-1, a_tup[-1]])
  a_n = tf.matmul(a_i, b)
  a_n = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
  return a_n


def ni_slice(sub_values, last_ind, axis=0):
  # TODO: Allow both to be negative indexed...
  ndims = len(shape(sub_values))
  im1 = 0 + abs(last_ind)
  i = [[None, None]] * ndims
  i[axis] = [im1, None]
  am = [False] * ndims
  am[axis] = True
  sl = [slice(*ii) for ii in i]
  ti = tf.reverse(sub_values, am)[sl]
  return tf.reverse(ti, am)


def ni(t, ind, axis=0):
  # Negative single index helper
  ndims = len(shape(t))
  im1 = -1 + abs(ind)
  i = [[None, None]] * ndims
  i[axis] = [im1, im1 + 1]
  am = [False] * ndims
  am[axis] = True
  sl = [slice(*ii) for ii in i]
  ti = tf.reverse(t, am)[sl]
  return ti[0, :, :]


def scan(fn, sequences, outputs_info):
  # for some reason TF step needs initializer passed as first argument?
  # a tiny wrapper to tf.scan to make my life easier
  # closer to theano scan, allows for step functions with multiple arguments
  # may eventually have kwargs which match theano
  for i in range(len(sequences)):
    # Try to accomodate for masks...
    seq = sequences[i]
    nd = ndim(seq)
    if nd == 3:
      pass
    elif nd < 3:
      sequences[i] = tf.expand_dims(sequences[i], nd)
    else:
      raise ValueError('Ndim too different to correct')

  def check(l):
    shapes = [shape(s) for s in l]
    # for now assume -1, can add axis argument later
    # check shapes match for concatenation
    compat = [ls for ls in shapes if ls[:-1] == shapes[0][:-1]]
    if len(compat) != len(shapes):
      raise ValueError('Tensors *must* be the same dim for now')

  check(sequences)
  check(outputs_info)

  seqs_shapes = [shape(s) for s in sequences]
  nd = len(seqs_shapes[0])
  seq_pack = tf.concat(nd - 1, sequences)
  outs_shapes = [shape(o) for o in outputs_info]
  nd = len(outs_shapes[0])
  init_pack = tf.concat(nd - 1, outputs_info)

  assert len(shape(seq_pack)) == 3
  assert len(shape(init_pack)) == 2

  def s_e(shps):
    starts = []
    ends = []
    prev_shp = 0
    for _, shp in enumerate(shps):
      start = prev_shp
      end = start + shp[-1]
      starts.append(start)
      ends.append(end)
      prev_shp = end
    return starts, ends

  # TF puts the initializer in front?
  def fnwrap(initializer, elems):
    starts, ends = s_e(seqs_shapes)
    sliced_elems = [elems[:, start:end] for start, end in zip(starts, ends)]
    starts, ends = s_e(outs_shapes)
    sliced_inits = [initializer[:, start:end]
                    for start, end in zip(starts, ends)]
    t = []
    t.extend(sliced_elems)
    t.extend(sliced_inits)
    # elems first then inits
    outs = fn(*t)
    nd = len(outs_shapes[0])
    outs_pack = tf.concat(nd - 1, outs)
    return outs_pack

  r = tf.scan(fnwrap, seq_pack, initializer=init_pack)

  if len(outs_shapes) > 1:
    starts, ends = s_e(outs_shapes)
    o = [r[:, :, start:end] for start, end in zip(starts, ends)]
    return o
  else:
    return r

##
# end metautils
##

##
# begin datasets
##


def duration_and_pitch_to_midi(filename, durations, pitches, prime_until=0):
  # durations and pitches should both be 2D
  # [time_steps, n_notes]

  sequence = music_pb2.NoteSequence()

  # Hardcode for now, eventually randomize?
  # or predict...
  sequence.ticks_per_quarter = mm.STANDARD_PPQ
  ts = sequence.time_signatures.add()
  ts.time = 1.0
  ts.numerator = 4
  ts.denominator = 4

  ks = sequence.key_signatures.add()
  ks.key = 0
  ks.mode = ks.MAJOR

  tempos = sequence.tempos.add()
  tempos.qpm = 120
  # ti.simultaneous_notes
  sn = 4

  # Translate durations from TIME_CLASSES indexes to TIME_CLASSES values.
  # First, find the data to change, then change it.
  # If we don't do this as separate steps, we'll end up changing some data
  # more than once.
  dt = copy.deepcopy(durations)
  time_class_indexes = []
  for i in range(len(TIME_CLASSES)):
    time_class_indexes.append(dt == i)
  for i, time in enumerate(TIME_CLASSES):
    dt[time_class_indexes[i]] = time

  delta_times = [dt[..., i] for i in range(sn)]
  end_times = [delta_times[i].cumsum(axis=0) for i in range(sn)]
  start_times = [end_times[i] - delta_times[i] for i in range(sn)]
  voices = [pitches[..., i] for i in range(sn)]

  midi_notes = []
  default_instrument = 0
  default_program = 0
  priming_instrument = 79
  priming_program = 79
  sequence.total_time = float(max([end_times[i][-1] for i in range(sn)]))

  assert len(delta_times[0]) == len(voices[0])
  for n in range(len(delta_times[0])):
    for i in range(len(voices)):
      # Hardcode 1 sample for now
      v = voices[i][n]
      s = start_times[i][n]
      e = end_times[i][n]
      if v != 0.:
        # Skip silence voices... for now
        # namedtuple?
        if n >= prime_until:
          midi_notes.append((default_instrument, default_program, v, s, e))
        else:
          midi_notes.append((priming_instrument, priming_program, v, s, e))
  for tup in midi_notes:
    sequence_note = sequence.notes.add()
    i = tup[0]
    p = tup[1]
    v = tup[2]
    s = tup[3]
    e = tup[4]
    sequence_note.instrument = int(i)
    sequence_note.program = int(p)
    sequence_note.pitch = int(v)
    sequence_note.velocity = int(127.)
    sequence_note.start_time = float(s)
    sequence_note.end_time = float(e)

  pretty_midi_object = mm.sequence_proto_to_pretty_midi(sequence)
  pretty_midi_object.write(filename)


class TFRecordDurationAndPitchIterator(object):

  def __init__(self, files_path, minibatch_size, start_index=0,
               stop_index=np.inf, make_mask=False,
               sequence_length=None,
               randomize=True):
    """Supports regular int, negative indexing, or float for stop_index."""
    reader = mm.note_sequence_io.note_sequence_record_iterator(files_path)
    all_ds = []
    all_ps = []
    self.note_classes = list(np.arange(88 + 1))  # + 1 for silence
    # set automatically
    # self.simultaneous_notes = int(max(np.sum(self._data, axis=0)))
    self.simultaneous_notes = 4
    time_classes_set = set(TIME_CLASSES)
    for ns in reader:
      notes = ns.notes
      st = np.array([n.start_time for n in notes]).astype('float32')
      et = np.array([n.end_time for n in notes]).astype('float32')
      pi = np.array([n.pitch for n in notes]).astype('float32')

      sample_times = sorted(list(set(st)))
      # go straight for pitch and delta time encoding
      sn = self.simultaneous_notes
      pitch_slices = [pi[st == sti][::-1] for sti in sample_times]
      # This monster fills in 0s so that array size is consistent
      pitch_slices = [p[:sn] if len(p) >= sn
                      else
                      np.concatenate((p, np.array([0.] * (sn - len(p)),
                                                  dtype='float32')))
                      for p in pitch_slices]
      start_slices = [st[st == sti] for sti in sample_times]
      end_slices = [et[st == sti] for sti in sample_times]
      start_slices = [ss[:sn] if len(ss) >= sn
                      else
                      np.concatenate((ss, np.array([ss[0]] * (sn - len(ss)),
                                                   dtype='float32')))
                      for ss in start_slices]
      end_slices = [es[:sn] if len(es) >= sn
                    else
                    np.concatenate((es, np.array([max(es)] * (sn - len(es)),
                                                 dtype='float32')))
                    for es in end_slices]
      start_slices = np.array(start_slices)
      end_slices = np.array(end_slices)
      delta_slices = end_slices - start_slices
      unsupported_time_classes = set(delta_slices.ravel()) - time_classes_set
      if unsupported_time_classes:
        tf.logging.warning(
            'NoteSequence %s:%s has unsupported time classes %s and will be '
            'skipped',
            ns.id, ns.filename, unsupported_time_classes)
        continue
      all_ds.append(np.array(delta_slices))
      all_ps.append(np.array(pitch_slices))
    assert len(all_ds) == len(all_ps)
    all_ds = np.concatenate(all_ds)
    all_ps = np.concatenate(all_ps)

    self._min_time_data = np.min(all_ds)
    self._max_time_data = np.max(all_ds)

    truncate = len(all_ds) - len(all_ds) % minibatch_size
    all_ds = all_ds[:truncate]
    all_ps = all_ps[:truncate]

    # transpose necessary to preserve data structure!
    # cut the audio into long contiguous subsequences based on the minibatch
    # size.
    all_ds = all_ds.transpose(1, 0)
    all_ds = all_ds.reshape(-1, minibatch_size,
                            all_ds.shape[1] // minibatch_size)
    all_ds = all_ds.transpose(2, 1, 0)
    all_ps = all_ps.transpose(1, 0)
    all_ps = all_ps.reshape(-1, minibatch_size,
                            all_ps.shape[1] // minibatch_size)
    all_ps = all_ps.transpose(2, 1, 0)

    len_ = len(all_ds)
    self._time_data = all_ds
    self._pitch_data = all_ps

    self.minibatch_size = minibatch_size
    self.sequence_length = sequence_length
    if randomize:
      self.random_state = np.random.RandomState(2177)
    self.make_mask = make_mask

    if stop_index >= 1:
      self.stop_index = int(min(stop_index, len_))
    elif stop_index > 0:
      # percentage
      self.stop_index = int(stop_index * len_)
    elif stop_index < 0:
      # negative index - must be int!
      self.stop_index = len_ + int(stop_index)

    self.start_index = start_index
    if start_index < 0:
      # negative indexing
      self.start_index = len_ + start_index
    elif start_index < 1:
      # float
      self.start_index = int(start_index * len_)
    else:
      # regular
      self.start_index = int(start_index)
    if self.start_index >= self.stop_index:
      ss = 'Invalid indexes - stop '
      ss += '%s <= start %s !' % (self.stop_index, self.start_index)
      raise ValueError(ss)
    self._current_index = self.start_index

  def __iter__(self):
    return self

  def next(self):
    return self.__next__()

  def __next__(self):
    s = self._current_index
    e = s + self.sequence_length
    if e > self.stop_index:
      raise StopIteration('End of file iterator reached!')
    time_data = np.array(self._time_data[s:e])

    # Translate durations from TIME_CLASSES values to TIME_CLASSES indexes.
    # First, find the data to change, then change it.
    # If we don't do this as separate steps, we'll end up changing some data
    # more than once.
    time_class_indexes = []
    for time in TIME_CLASSES:
      time_class_indexes.append(time_data == time)
    for i in range(len(TIME_CLASSES)):
      time_data[time_class_indexes[i]] = i

    pitch_data = np.array(self._pitch_data[s:e])

    if self.make_mask is False:
      res = (time_data, pitch_data)
    else:
      raise ValueError('Unhandled mask making')
    self._current_index = e
    return res

  def reset(self):
    self._current_index = self.start_index

##
# end datasets
##

##
# begin initializers and Theano functions
##


def np_zeros(shp):
  """Builds a numpy variable filled with zeros.

  Args:
    shp: tuple of ints
      shape of zeros to initialize

  Returns:
    initialized_zeros, array-like
      Array-like of zeros the same size as shape parameter
  """
  return np.zeros(shp).astype('float32')


def np_normal(shp, random_state, scale=0.01):
  """Builds a numpy variable filled with normal random values.

  Args:
    shp: tuple of ints or tuple of tuples
      shape of values to initialize
      tuple of ints should be single shape
      tuple of tuples is primarily for convnets and should be of form
      ((n_in_kernels, kernel_width, kernel_height),
       (n_out_kernels, kernel_width, kernel_height))
    random_state: numpy.random.RandomState() object
    scale: float (default 0.01)
      default of 0.01 results in normal random values with variance 0.01

  Returns:
    initialized_normal, array-like
      Array-like of normal random values the same size as shape parameter
  """
  if isinstance(shp[0], tuple):
    shp = (shp[1][0], shp[0][0]) + shp[1][1:]
  return (scale * random_state.randn(*shp)).astype('float32')


def np_tanh_fan_normal(shp, random_state, scale=1.):
  """Builds a numpy variable filled with random values.

  Args:
    shp: tuple of ints or tuple of tuples
      shape of values to initialize
      tuple of ints should be single shape
      tuple of tuples is primarily for convnets and should be of form
      ((n_in_kernels, kernel_width, kernel_height),
       (n_out_kernels, kernel_width, kernel_height))
    random_state: numpy.random.RandomState() object
    scale: float (default 1.)
      default of 1. results in normal random values
      with sqrt(2 / (fan in + fan out)) scale

  Returns:
    initialized_fan, array-like
      Array-like of random values the same size as shape parameter

  References
  ----------
  Understanding the difficulty of training deep feedforward neural networks
      X. Glorot, Y. Bengio
  """
  # The . after the 2 is critical! shape has dtype int...
  if isinstance(shp[0], tuple):
    kern_sum = np.prod(shp[0]) + np.prod(shp[1])
    shp = (shp[1][0], shp[0][0]) + shp[1][1:]
  else:
    kern_sum = np.sum(shp)
  var = scale * np.sqrt(2. / kern_sum)
  return var * random_state.randn(*shp).astype('float32')


def np_variance_scaled_uniform(shp, random_state, scale=1.):
  """Builds a numpy variable filled with random values.

  Args:
    shp: tuple of ints or tuple of tuples
      shape of values to initialize
      tuple of ints should be single shape
      tuple of tuples is primarily for convnets and should be of form
      ((n_in_kernels, kernel_width, kernel_height),
       (n_out_kernels, kernel_width, kernel_height))
    random_state: numpy.random.RandomState() object
    scale: float (default 1.)
      default of 1. results in uniform random values
      with 1 * sqrt(1 / (n_dims)) scale

  Returns:
    initialized_scaled, array-like
      Array-like of random values the same size as shape parameter

  References
  ----------
  Efficient Backprop
      Y. LeCun, L. Bottou, G. Orr, K. Muller
  """
  if isinstance(shp[0], tuple):
    shp = (shp[1][0], shp[0][0]) + shp[1][1:]
    kern_sum = np.prod(shp[0])
  else:
    kern_sum = shp[0]
  #  Make sure bounds aren't the same
  bound = scale * np.sqrt(3. / kern_sum)  # sqrt(3) for std of uniform
  return random_state.uniform(low=-bound, high=bound, size=shp).astype(
      'float32')


def np_ortho(shp, random_state, scale=1.):
  """Builds a numpy variable filled with orthonormal random values.

  Args:
    shp: tuple of ints or tuple of tuples
      shape of values to initialize
      tuple of ints should be single shape
      tuple of tuples is primarily for convnets and should be of form
      ((n_in_kernels, kernel_width, kernel_height),
       (n_out_kernels, kernel_width, kernel_height))
    random_state: numpy.random.RandomState() object
    scale: float (default 1.)
      default of 1. results in orthonormal random values sacled by 1.

  Returns:
    initialized_ortho, array-like
      Array-like of random values the same size as shape parameter

  References
  ----------
  Exact solutions to the nonlinear dynamics of learning in deep linear
  neural networks
      A. Saxe, J. McClelland, S. Ganguli
  """
  if isinstance(shp[0], tuple):
    shp = (shp[1][0], shp[0][0]) + shp[1][1:]
    flat_shp = (shp[0], np.prd(shp[1:]))
  else:
    flat_shp = shp
  g = random_state.randn(*flat_shp)
  u, _, vt = linalg.svd(g, full_matrices=False)
  res = u if u.shape == flat_shp else vt  # pick one with the correct shape
  res = res.reshape(shp)
  return (scale * res).astype('float32')


def make_numpy_biases(bias_dims):
  return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale='default'):
  """
  Will return as many things as are in the list of out_dims
  You *must* get a list back, even for 1 element retuTrue
  blah, = make_weights(...)
  or
  [blah] = make_weights(...)
  """
  ff = [None] * len(out_dims)
  for i, out_dim in enumerate(out_dims):
    if init is None:
      if in_dim == out_dim:
        ff[i] = np_ortho
      else:
        ff[i] = np_variance_scaled_uniform
    else:
      raise ValueError('Unknown init type %s' % init)
  if scale == 'default':
    ws = [ff[i]((in_dim, out_dim), random_state)
          for i, out_dim in enumerate(out_dims)]
  else:
    ws = [ff[i]((in_dim, out_dim), random_state, scale=scale)
          for i, out_dim in enumerate(out_dims)]
  return ws


# Storage of internal shared
_lib_shared_params = OrderedDict()


def _get_name():
  return str(uuid.uuid4())


def _get_shared(name):
  if name in _lib_shared_params.keys():
    tf.logging.info('Found name %s in shared parameters' % name)
    return _lib_shared_params[name]
  else:
    raise NameError('Name not found in shared params!')


def _set_shared(name, variable):
  if name in _lib_shared_params.keys():
    raise ValueError('Trying to set key %s which already exists!' % name)
  _lib_shared_params[name] = variable


def embedding(indices, n_symbols, output_dim, random_state, name=None):
  """Last dimension of indices tensor must be 1!!!!"""
  if name is None:
    name = _get_name()

  try:
    vectors = _get_shared(name)
  except NameError:
    vectors = tf.Variable(
        random_state.randn(n_symbols, output_dim).astype('float32'),
        trainable=True)
    _set_shared(name, vectors)

  ii = tf.cast(indices, 'int32')
  shp = shape(ii)
  nd = len(shp)
  lu = tf.nn.embedding_lookup(vectors, ii)
  if nd == 3:
    lu = lu[:, :, 0, :]
  else:
    return lu
  return lu


def multiembedding(multi_indices, n_symbols, output_dim, random_state,
                   name=None, share_all=False):
  """Helper to compute many embeddings and concatenate.

  Requires input indices to be 3D, with last axis being the "iteration"
  dimension
  """
  # Should n_symbols be a list of embedding values?
  output_embeds = []
  shp = shape(multi_indices)
  if len(shp) != 3:
    raise ValueError('Unhandled rank != 3 for input multi_indices')
  index_range = shp[-1]
  if share_all:
    if name is None:
      n = _get_name()
      names = [n] * index_range
    else:
      names = [name + '_0' for i in range(index_range)]
  else:
    if name is None:
      names = [_get_name() for i in range(index_range)]
    else:
      names = [name + '_%i' % i for i in range(index_range)]
  for i in range(index_range):
    e = embedding(multi_indices[:, :, i], n_symbols, output_dim, random_state,
                  name=names[i])
    output_embeds.append(e)
  return tf.concat(2, output_embeds)


def automask(input_tensor, n_masks, axis=-1):
  """Auto masker to make multiple MADE/pixelRNN style masking easier.

  n_masks *must* be an even divisor of input_tensor.shape[axis]

  masks are basically of form

  [:, :, :i * divisor_dim] = 1.
  for i in range(n_masks)

  a 1, 4 example with n_masks = 2 would be

  mask0 = [0., 0., 0., 0.]
  mask1 = [1., 1., 0., 0.]

  The goal of these masks is to model p(y_i,t | x_<=t, y_<i,<=t) in order
  to maximize the context a given prediction can "see".

  This function will return n_masks copies of input_tensor (* a mask) in list
  """
  if axis != -1:
    raise ValueError('Axis not currently supported!')
  shp = shape(input_tensor)
  shp_tup = tuple([1] * (len(shp) - 1) + [shp[-1]])
  div = shp[-1] // n_masks
  assert int(div * n_masks) == shp[-1]
  masks = [np.zeros(shp_tup).astype('float32') for i in range(n_masks)]
  if n_masks < 2:
    raise ValueError('unhandled small n_masks value')
  output_tensors = [masks[0] * input_tensor]
  for i in range(1, n_masks):
    masks[i][..., :i * div] = 1.
    output_tensors.append(masks[i] * input_tensor)
  return output_tensors


def linear(list_of_inputs, input_dims, output_dim, random_state, name=None,
           init=None, scale='default', weight_norm=None, biases=True):
  """Can pass weights and biases directly if needed through init."""
  if weight_norm is None:
    # Let other classes delegate to default of linear
    weight_norm = True
  # assume both have same shape ...
  nd = ndim(list_of_inputs[0])
  input_var = tf.concat(concat_dim=nd - 1, values=list_of_inputs)
  input_dim = sum(input_dims)
  terms = []
  if (init is None) or isinstance(init, str):
    weight_values, = make_numpy_weights(input_dim, [output_dim],
                                        random_state=random_state,
                                        init=init, scale=scale)
  else:
    weight_values = init[0]

  if name is None:
    name = _get_name()
  elif name[0] is None:
    name = (_get_name(),) + name[1:]
    name = '_'.join(name)

  name_w = name + '_linear_w'
  name_b = name + '_linear_b'
  name_wn = name + '_linear_wn'

  try:
    weight = _get_shared(name_w)
  except NameError:
    weight = tf.Variable(weight_values, trainable=True)
    _set_shared(name_w, weight)

  # Weight normalization... Kingma and Salimans
  # http://arxiv.org/abs/1602.07868
  if weight_norm:
    norm_values = np.linalg.norm(weight_values, axis=0)
    try:
      norms = _get_shared(name_wn)
    except NameError:
      norms = tf.Variable(norm_values, trainable=True)
      _set_shared(name_wn, norms)
    norm = tf.sqrt(tf.reduce_sum(tf.abs(weight ** 2), reduction_indices=[0],
                                 keep_dims=True))
    normed_weight = weight * (norms / norm)
    terms.append(dot(input_var, normed_weight))
  else:
    terms.append(dot(input_var, weight))

  if biases:
    if (init is None) or isinstance(init, str):
      b, = make_numpy_biases([output_dim])
    else:
      b = init[1]
    try:
      biases = _get_shared(name_b)
    except NameError:
      biases = tf.Variable(b, trainable=True)
      _set_shared(name_b, biases)
    terms.append(biases)
  out = sum(terms)
  return out


def gru_weights(input_dim, hidden_dim, forward_init=None, hidden_init='normal',
                random_state=None):
  if random_state is None:
    raise ValueError('Must pass random_state!')
  shp = (input_dim, hidden_dim)
  if forward_init == 'normal':
    w = np.hstack([np_normal(shp, random_state),
                   np_normal(shp, random_state),
                   np_normal(shp, random_state)])
  elif forward_init == 'fan':
    w = np.hstack([np_tanh_fan_normal(shp, random_state),
                   np_tanh_fan_normal(shp, random_state),
                   np_tanh_fan_normal(shp, random_state)])
  elif forward_init is None:
    if input_dim == hidden_dim:
      w = np.hstack([np_ortho(shp, random_state),
                     np_ortho(shp, random_state),
                     np_ortho(shp, random_state)])
    else:
      # lecun
      w = np.hstack([np_variance_scaled_uniform(shp, random_state),
                     np_variance_scaled_uniform(shp, random_state),
                     np_variance_scaled_uniform(shp, random_state)])
  else:
    raise ValueError('Unknown forward init type %s' % forward_init)
  b = np_zeros((3 * shp[1],))

  if hidden_init == 'normal':
    wur = np.hstack([np_normal((shp[1], shp[1]), random_state),
                     np_normal((shp[1], shp[1]), random_state)])
    u = np_normal((shp[1], shp[1]), random_state)
  elif hidden_init == 'ortho':
    wur = np.hstack([np_ortho((shp[1], shp[1]), random_state),
                     np_ortho((shp[1], shp[1]), random_state)])
    u = np_ortho((shp[1], shp[1]), random_state)
  return w, b, wur, u


def gru(inp, gate_inp, previous_state, input_dim, hidden_dim, random_state,
        mask=None, name=None, init=None, weight_norm=None,
        biases=False):
  if name is not None:
    raise ValueError('Unhandled parameter sharing in GRU')
  if init is None:
    hidden_init = 'ortho'
  elif init == 'normal':
    hidden_init = 'normal'
  else:
    raise ValueError('Not yet configured for other inits')

  ndi = ndim(inp)
  if mask is None:
    if ndi == 2:
      mask = tf.ones_like(inp)
    else:
      raise ValueError('Unhandled ndim')

  ndm = ndim(mask)
  if ndm == (ndi - 1):
    mask = tf.expand_dims(mask, ndm - 1)

  _, _, wur, u = gru_weights(input_dim, hidden_dim,
                             hidden_init=hidden_init,
                             random_state=random_state)
  dim = hidden_dim
  f1 = linear([previous_state], [2 * hidden_dim], 2 * hidden_dim,
              random_state, name=(name, 'update/reset'), init=[wur],
              biases=biases, weight_norm=weight_norm)
  gates = sigmoid(f1 + gate_inp)
  update = gates[:, :dim]
  reset = gates[:, dim:]
  state_reset = previous_state * reset
  f2 = linear([state_reset], [hidden_dim], hidden_dim,
              random_state, name=(name, 'state'), init=[u], biases=biases,
              weight_norm=weight_norm)
  next_state = tf.tanh(f2 + inp)
  next_state = next_state * update + previous_state * (1. - update)
  next_state = mask * next_state + (1. - mask) * previous_state
  return next_state


def gru_fork(list_of_inputs, input_dims, output_dim, random_state, name=None,
             init=None, scale='default', weight_norm=None, biases=True):
  if name is not None:
    raise ValueError('Unhandled parameter sharing in gru_fork')
  gates = linear(list_of_inputs, input_dims, 3 * output_dim,
                 random_state=random_state,
                 name=(name, 'gates'), init=init, scale=scale,
                 weight_norm=weight_norm, biases=biases)
  dim = output_dim
  nd = ndim(gates)
  if nd == 2:
    d = gates[:, :dim]
    g = gates[:, dim:]
  elif nd == 3:
    d = gates[:, :, :dim]
    g = gates[:, :, dim:]
  else:
    raise ValueError('Unsupported ndim')
  return d, g


def lstm_weights(input_dim, hidden_dim, forward_init=None, hidden_init='normal',
                 random_state=None):
  if random_state is None:
    raise ValueError('Must pass random_state!')
  shp = (input_dim, hidden_dim)
  if forward_init == 'normal':
    w = np.hstack([np_normal(shp, random_state),
                   np_normal(shp, random_state),
                   np_normal(shp, random_state),
                   np_normal(shp, random_state)])
  elif forward_init == 'fan':
    w = np.hstack([np_tanh_fan_normal(shp, random_state),
                   np_tanh_fan_normal(shp, random_state),
                   np_tanh_fan_normal(shp, random_state),
                   np_tanh_fan_normal(shp, random_state)])
  elif forward_init is None:
    if input_dim == hidden_dim:
      w = np.hstack([np_ortho(shp, random_state),
                     np_ortho(shp, random_state),
                     np_ortho(shp, random_state),
                     np_ortho(shp, random_state)])
    else:
      # lecun
      w = np.hstack([np_variance_scaled_uniform(shp, random_state),
                     np_variance_scaled_uniform(shp, random_state),
                     np_variance_scaled_uniform(shp, random_state),
                     np_variance_scaled_uniform(shp, random_state)])
  else:
    raise ValueError('Unknown forward init type %s' % forward_init)
  b = np_zeros((4 * shp[1],))
  # Set forget gate bias to 1
  b[shp[1]:2 * shp[1]] += 1.

  if hidden_init == 'normal':
    u = np.hstack([np_normal((shp[1], shp[1]), random_state),
                   np_normal((shp[1], shp[1]), random_state),
                   np_normal((shp[1], shp[1]), random_state),
                   np_normal((shp[1], shp[1]), random_state)])
  elif hidden_init == 'ortho':
    u = np.hstack([np_ortho((shp[1], shp[1]), random_state),
                   np_ortho((shp[1], shp[1]), random_state),
                   np_ortho((shp[1], shp[1]), random_state),
                   np_ortho((shp[1], shp[1]), random_state)])
  return w, b, u


def lstm(inp, gate_inp, previous_state, input_dim, hidden_dim, random_state,
         mask=None, name=None, init=None):
  """
  Output is the concatenation of hidden state and cell
  so 2 * hidden dim
  will need to slice yourself, or handle in some way
  This was done specifically to have the GRU, LSTM activations swappable
  """
  if name is not None:
    raise ValueError('Unhandled parameter sharing in LSTM')
  if gate_inp != 'LSTMGates':
    raise ValueError('Use lstm_fork to setup this block')
  if init is None:
    hidden_init = 'ortho'
  elif init == 'normal':
    hidden_init = 'normal'
  else:
    raise ValueError('Not yet configured for other inits')

  ndi = ndim(inp)
  if mask is None:
    if ndi == 2:
      mask = tf.ones_like(inp)[:, :hidden_dim]
    else:
      raise ValueError('Unhandled ndim')

  ndm = ndim(mask)
  if ndm == (ndi - 1):
    mask = tf.expand_dims(mask, ndm - 1)

  _, _, u = lstm_weights(input_dim, hidden_dim,
                         hidden_init=hidden_init,
                         random_state=random_state)
  dim = hidden_dim

  def _s(p, d):
    return p[:, d * dim:(d+1) * dim]

  previous_cell = _s(previous_state, 1)
  previous_st = _s(previous_state, 0)

  preactivation = linear([previous_st], [4 * hidden_dim],
                         4 * hidden_dim,
                         random_state, name=(name, 'preactivation'),
                         init=[u],
                         biases=False) + inp

  ig = sigmoid(_s(preactivation, 0))
  fg = sigmoid(_s(preactivation, 1))
  og = sigmoid(_s(preactivation, 2))
  cg = tanh(_s(preactivation, 3))

  cg = fg * previous_cell + ig * cg
  cg = mask * cg + (1. - mask) * previous_cell

  hg = og * tanh(cg)
  hg = mask * hg + (1. - mask) * previous_st

  next_state = tf.concat(1, [hg, cg])
  return next_state


def lstm_fork(list_of_inputs, input_dims, output_dim, random_state, name=None,
              scale='default', weight_norm=None):
  """
  output dim should be the hidden size for each gate
  overall size will be 4x
  """
  if name is not None:
    raise ValueError('Unhandled parameter sharing in lstm_fork')
  inp_d = np.sum(input_dims)
  w, b, _ = lstm_weights(inp_d, output_dim,
                         random_state=random_state)
  f_init = [w, b]
  inputs = linear(list_of_inputs, input_dims, 4 * output_dim,
                  random_state=random_state,
                  name=(name, 'inputs'), init=f_init, scale=scale,
                  weight_norm=weight_norm,
                  biases=True)
  return inputs, 'LSTMGates'


def softmax(x):
  # should work for both 2D and 3D
  dim = len(shape(x))
  e_x = tf.exp(x - tf.reduce_max(x, reduction_indices=[dim - 1],
                                 keep_dims=True))
  out = e_x / tf.reduce_sum(e_x, reduction_indices=[dim - 1], keep_dims=True)
  return out


def numpy_softmax(x, temperature=1.):
  # should work for both 2D and 3D
  dim = x.ndim
  x /= temperature
  e_x = np.exp((x - x.max(axis=dim - 1, keepdims=True)))
  out = e_x / e_x.sum(axis=dim - 1, keepdims=True)
  return out


def tanh(x):
  return tf.tanh(x)


def sigmoid(x):
  return tf.sigmoid(x)


def categorical_crossentropy(predicted_values, true_values, class_weights=None,
                             eps=None):
  """Categorical crossentropy.

  Multinomial negative log likelihood of predicted compared to one hot
  true_values

  Args:
    predicted_values: tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of a softmax
    true_values: tensor, shape 2D or 3D
        Ground truth one hot values
    eps: float, default None
        Epsilon to be added during log calculation to avoid NaN values.
    class_weights: dictionary with form {class_index: weight)
        Unspecified classes will get the default weight of 1.
        See discussion here for understanding how class weights work
        http://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work

  Returns:
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
      The cost per sample, or per sample per step if 3D
  """
  if eps is not None:
    raise ValueError('Not yet implemented')
  else:
    predicted_values = tf.to_float(predicted_values)
    true_values = tf.to_float(true_values)
  tshp = shape(true_values)
  pshp = shape(predicted_values)
  if tshp[-1] == 1 or len(tshp) < len(pshp):
    tf.logging.info('True values dimension should match predicted!')
    tf.logging.info('Expected %s, got %s' % (pshp, tshp))
    if tshp[-1] == 1:
      # squeeze out the last dimension
      tf.logging.info('Removing last dimension of 1 from %s' % str(tshp))
      if len(tshp) == 3:
        true_values = true_values[:, :, 0]
      elif len(tshp) == 2:
        true_values = true_values[:, 0]
      else:
        raise ValueError('Unhandled dimensions in squeeze')
    tshp = shape(true_values)
    if len(tshp) == (len(pshp) - 1):
      tf.logging.info('Changing %s to %s with one hot encoding' % (tshp, pshp))
      tf.cast(true_values, 'int32')
      ot = tf.one_hot(tf.cast(true_values, 'int32'), pshp[-1],
                      dtype='float32', axis=-1)
      true_values = ot
    elif len(tshp) == len(pshp):
      pass
    else:
      raise ValueError('Dimensions of true_values and predicted_values'
                       'mismatched')
    # repeat so the right shape is captured
    tshp = shape(true_values)
  cw = np.ones(pshp[-1], dtype='float32')
  if class_weights is not None:
    for k, v in class_weights.items():
      cw[k] = v
    cw /= np.sum(cw)
    # np.sum() cw really should be close to 1
    cw /= np.sum(cw) + 1E-12
  # expand dimensions for broadcasting
  if len(tshp) == 3:
    cw = cw[None, None, :]
  elif len(tshp) == 2:
    cw = cw[None, :]
  nd = len(shape(true_values))
  assert nd == len(shape(predicted_values))
  stable_result = tf.select(true_values < 1E-20, 0. * predicted_values,
                            cw * true_values * tf.log(predicted_values))
  ce = -tf.reduce_sum(stable_result, reduction_indices=[nd - 1])
  return ce


def numpy_sample_softmax(coeff, random_state, class_weights=None, debug=False):
  """Numpy function to sample from a softmax distribution.

  Args:
  coeff: array-like, shape 2D or higher
      The predicted class probabilities out of some layer,
      normally the output of a softmax
  random_state: numpy.random.RandomState() instance
  class_weights: dictionary with form {class_index: weight}, default None
      Unspecified classes will get the default weight of 1.
      See discussion here for understanding how class weights work
      http://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work
  debug: Boolean, default False
      Take the argmax instead of sampling. Useful for debugging purposes or
      testing greedy sampling.

  Returns:
    samples : array-like, shape of coeff.shape[:-1]
      Sampled values
  """
  reshape_dims = coeff.shape[:-1]
  coeff = coeff.reshape((-1, coeff.shape[-1]))
  cw = np.ones((1, coeff.shape[-1])).astype('float32')
  if class_weights is not None:
    for k, v in class_weights.items():
      cw[k] = v
    cw /= np.sum(cw)
    cw /= np.sum(cw) + 1E-12
  if debug:
    idx = coeff.argmax(axis=-1)
  else:
    coeff = cw * coeff
    # renormalize to avoid numpy errors about summation...
    # end result shouldn't change
    coeff /= coeff.sum(axis=1, keepdims=True) + 1E-3
    idxs = [np.argmax(random_state.multinomial(1, pvals=coeff[i]))
            for i in range(len(coeff))]
    idx = np.array(idxs)
  idx = idx.reshape(reshape_dims)
  return idx.astype('float32')

##
# end initializers and Theano functions
##

##
# start training utilities
##


def save_checkpoint(checkpoint_dir, checkpoint_name, saver, sess):
  checkpoint_save_path = os.path.join(checkpoint_dir, checkpoint_name)
  saver.save(sess, checkpoint_save_path)
  tf.logging.info('Model saved to %s' % checkpoint_save_path)


def run_loop(loop_function, train_dir, train_itr, valid_itr, n_epochs,
             checkpoint_delay=10, checkpoint_every_n_epochs=1,
             checkpoint_every_n_updates=np.inf,
             skip_minimums=False,
             skip_n_train_minibatches=-1):
  """Loop function.

  loop function must have the following api
  loop(itr, sess, inits=None, do_updates=True)
        return cost, init_1, init_2, ....
  must pass back a list!!! For only output cost, do
      return [cost]
  do_updates will control what happens in a validation loop
  inits will pass init_1, init_2,  ... back into the loop
  loop function should return a list of [cost] + all_init_hiddens or other
  states
  """
  tf.logging.info('Running loops...')
  loop = loop_function

  checkpoint_dict = {}
  overall_train_costs = []
  overall_valid_costs = []
  overall_train_checkpoint = []
  overall_valid_checkpoint = []
  start_epoch = 0

  # If there are more than 1M minibatches per epoch this will break!
  # Not reallocating buffer greatly helps fast training models though
  # We have bigger problems if there are 1M minibatches per epoch...
  # This will get sliced down to the correct number of minibatches
  # During calculations down below
  train_costs = [0.] * 1000000
  valid_costs = [0.] * 1000000
  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    av = tf.all_variables()
    train_saver = tf.train.Saver(av)
    valid_saver = tf.train.Saver(av)
    force_saver = tf.train.Saver(av)
    try:
      for e in range(start_epoch, start_epoch + n_epochs):
        tf.logging.info('Starting training, epoch %i' % e)
        train_mb_count = 0
        valid_mb_count = 0
        results_dict = {k: v for k, v in checkpoint_dict.items()}
        this_results_dict = results_dict
        try:
          # Start training
          inits = None
          train_itr.reset()
          while True:
            if train_mb_count < skip_n_train_minibatches:
              next(train_itr)
              train_mb_count += 1
              continue
            r = loop(train_itr, sess, inits=inits, do_updates=True)
            partial_train_costs = r[0]
            if len(r) > 1:
              inits = r[1:]
            else:
              # Special case of 1 output [cost]
              pass
            train_costs[train_mb_count] = np.mean(partial_train_costs)
            tc = train_costs[train_mb_count]
            train_mb_count += 1
            if np.isnan(tc):
              tf.logging.info(
                  'NaN detected in train cost, update %i' % train_mb_count)
              raise ValueError('NaN detected in train')

            if (train_mb_count % checkpoint_every_n_updates) == 0:
              checkpoint_save_path = (
                  'model_update_checkpoint_%i.ckpt' % train_mb_count)
              save_checkpoint(
                  train_dir, checkpoint_save_path, train_saver, sess)

              tf.logging.info(
                  'Update checkpoint after train mb %i' % train_mb_count)
              tf.logging.info(
                  'Current mean cost %f' % np.mean(partial_train_costs))

              this_results_dict['this_epoch_train_auto'] = (
                  train_costs[:train_mb_count])
              tmb = train_costs[:train_mb_count]
              running_train_mean = np.cumsum(tmb) / (
                  np.arange(train_mb_count) + 1)
              # needs to be a list
              running_train_mean = list(running_train_mean)
              this_results_dict['this_epoch_train_mean_auto'] = (
                  running_train_mean)
        except StopIteration:
          # Slice so that only valid data is in the minibatch
          # this also assumes there is not a variable number
          # of minibatches in an epoch!
          # edge case - add one since stop iteration was raised
          # before increment
          train_costs_slice = train_costs[:train_mb_count + 1]

          # Start validation
          tf.logging.info('Starting validation, epoch %i' % e)
          inits = None
          valid_itr.reset()
          try:
            while True:
              r = loop(valid_itr, sess, inits=inits, do_updates=False)
              partial_valid_costs = r[0]
              if len(r) > 1:
                inits = r[1:]
              else:
                pass
              valid_costs[valid_mb_count] = np.mean(partial_valid_costs)
              vc = valid_costs[valid_mb_count]
              valid_mb_count += 1
              if np.isnan(vc):
                tf.logging.info(
                    'NaN detected in valid cost, minibatch %i' % valid_mb_count)
                raise ValueError('NaN detected in valid')
          except StopIteration:
            # Hit end of iterator
            pass
          # edge case - add one since stop iteration was raised
          # before increment
          valid_costs_slice = valid_costs[:valid_mb_count + 1]

          mean_epoch_train_cost = np.mean(train_costs_slice)
          # np.inf trick to avoid taking the min of length 0 list
          old_min_train_cost = min(overall_train_costs + [np.inf])
          if np.isnan(mean_epoch_train_cost):
            tf.logging.info(
                'Previous train costs %s' % overall_train_costs[-5:])
            tf.logging.info('NaN detected in train cost, epoch %i' % e)
            raise ValueError('NaN detected in train')
          overall_train_costs.append(mean_epoch_train_cost)

          mean_epoch_valid_cost = np.mean(valid_costs_slice)
          old_min_valid_cost = min(overall_valid_costs + [np.inf])
          if np.isnan(mean_epoch_valid_cost):
            tf.logging.info(
                'Previous valid costs %s' % overall_valid_costs[-5:])
            tf.logging.info('NaN detected in valid cost, epoch %i' % e)
            raise ValueError('NaN detected in valid')
          overall_valid_costs.append(mean_epoch_valid_cost)

          if mean_epoch_train_cost < old_min_train_cost:
            overall_train_checkpoint.append(mean_epoch_train_cost)
          else:
            overall_train_checkpoint.append(old_min_train_cost)

          if mean_epoch_valid_cost < old_min_valid_cost:
            overall_valid_checkpoint.append(mean_epoch_valid_cost)
          else:
            overall_valid_checkpoint.append(old_min_valid_cost)

          checkpoint_dict['train_costs'] = overall_train_costs
          checkpoint_dict['valid_costs'] = overall_valid_costs
          # Tracking if checkpoints are made
          checkpoint_dict['train_checkpoint_auto'] = overall_train_checkpoint
          checkpoint_dict['valid_checkpoint_auto'] = overall_valid_checkpoint

          tf.logging.info('Epoch %i complete' % e)
          tf.logging.info('Epoch mean train cost %f' % mean_epoch_train_cost)
          tf.logging.info('Epoch mean valid cost %f' % mean_epoch_valid_cost)
          tf.logging.info('Previous train costs %s' % overall_train_costs[-5:])
          tf.logging.info('Previous valid costs %s' % overall_valid_costs[-5:])

          results_dict = {k: v for k, v in checkpoint_dict.items()}

          if e < checkpoint_delay or skip_minimums:
            pass
          elif mean_epoch_valid_cost < old_min_valid_cost:
            tf.logging.info('Checkpointing valid...')
            checkpoint_save_path = 'model_checkpoint_valid_%i.ckpt' % e
            save_checkpoint(train_dir, checkpoint_save_path, valid_saver, sess)
            tf.logging.info('Valid checkpointing complete.')
          elif mean_epoch_train_cost < old_min_train_cost:
            tf.logging.info('Checkpointing train...')
            checkpoint_save_path = 'model_checkpoint_train_%i.ckpt' % e
            save_checkpoint(train_dir, checkpoint_save_path, train_saver, sess)
            tf.logging.info('Train checkpointing complete.')

          if e < checkpoint_delay:
            pass
            # Don't skip force checkpoints after default delay
            # Printing already happens above
          elif((e % checkpoint_every_n_epochs) == 0) or (e == (n_epochs - 1)):
            tf.logging.info('Checkpointing force...')
            checkpoint_save_path = 'model_checkpoint_%i.ckpt' % e
            save_checkpoint(train_dir, checkpoint_save_path, force_saver, sess)
            tf.logging.info('Force checkpointing complete.')
    except KeyboardInterrupt:
      tf.logging.info('Training loop interrupted by user!')
  tf.logging.info(
      'Loop finished, closing write threads (this may take a while!)')

##
# end training utilities
##
