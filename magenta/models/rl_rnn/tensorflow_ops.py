import tensorflow as tf

def make_cell(hparams, state_is_tuple=False):
  cells = []
  for num_units in hparams.rnn_layer_sizes:
    cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units, state_is_tuple=state_is_tuple)
    #cell = tf.nn.rnn_cell.DropoutWrapper(
    #    cell, output_keep_prob=hparams.dropout_keep_prob)
    cells.append(cell)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if hparams.attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, hparams.attn_length, state_is_tuple=state_is_tuple)

  return cell

def old_make_cell_according_to_fjord(hparams):
  lstm_layers = [ tf.nn.rnn_cell.LSTMCell(
    num_units=layer_size, state_is_tuple=False) for layer_size in hparams.rnn_layer_sizes ] 
  multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers, state_is_tuple=False) 
  return multi_cell