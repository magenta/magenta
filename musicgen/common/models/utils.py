import tensorflow as tf

# Stolen from magenta/models/shared/events_rnn_graph
def make_rnn_cell(rnn_layer_sizes,
				  dropout_keep_prob=1.0,
				  attn_length=0,
				  base_cell=tf.contrib.rnn.BasicLSTMCell,
				  state_is_tuple=False):
  cells = []
  for num_units in rnn_layer_sizes:
	cell = base_cell(num_units, state_is_tuple=state_is_tuple)
	cell = tf.contrib.rnn.DropoutWrapper(
		cell, output_keep_prob=dropout_keep_prob)
	cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
	cell = tf.contrib.rnn.AttentionCellWrapper(
		cell, attn_length, state_is_tuple=state_is_tuple)

  return cell