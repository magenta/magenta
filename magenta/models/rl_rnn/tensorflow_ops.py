import tensorflow as tf

def make_cell(hparams):
  cells = []
  for num_units in hparams.rnn_layer_sizes:
    cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units, state_is_tuple=state_is_tuple)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, output_keep_prob=hparams.dropout_keep_prob)
    cells.append(cell)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if hparams.attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, hparams.attn_length, state_is_tuple=state_is_tuple)

  return cell