This directory contains code and content that is shared by multiple models.

The primer.mid MIDI file is an original three bar long monophonic melody. It can be used as a priming melody when generating melodies with any of the melody RNN models. Example usage:

```
bazel build magenta/models/basic_rnn:basic_rnn_generate

./bazel-bin/magenta/models/basic_rnn/basic_rnn_generate \
--run_dir=/tmp/basic_rnn/logdir/run1 \
--output_dir=/tmp/basic_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_midi=<absolute path to primer.mid>
```
