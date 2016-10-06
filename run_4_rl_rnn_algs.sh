bazel run magenta/models/rl_rnn:melody_q_train -- --algorithm 'ftq'
bazel run magenta/models/rl_rnn:melody_q_train -- --algorithm 'g'
bazel run magenta/models/rl_rnn:melody_q_train -- --algorithm 'psi'
bazel run magenta/models/rl_rnn:melody_q_train -- --algorithm 'pure_rl'
