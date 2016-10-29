py_binary(
    name = "melody_q_train",
    srcs = ["melody_q_train.py"],
    deps = [
        ":melody_q",
        ":rl_rnn_ops",
        "//magenta/common:tf_lib",
    ],
)

py_binary(
    name = "melody_q",
    srcs = ["melody_q.py"],
    deps = [
        ":melody_rnn",
        ":rl_rnn_ops",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn",
    srcs = ["melody_rnn.py"],
    deps = [
        ":melody_rnn_encoder_decoder",
        ":rl_rnn_ops",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        "//magenta/music:sequences_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn_encoder_decoder",
    srcs = ["melody_rnn_encoder_decoder.py"],
    deps = [
        "//magenta/music:melodies_lib",
    ],
)

py_binary(
    name = "rl_rnn_ops",
    srcs = ["rl_rnn_ops.py"],
    deps = [
        "//magenta/common:tf_lib",
        # tensorflow dep
    ],
)

py_test(
    name = "melody_q_test",
    size = "medium",
    srcs = ["melody_q_test.py"],
    data =
        ["testdata/primer.mid"] + glob(["testdata/train/*"]),
    deps = [
        ":melody_q",
        ":rl_rnn_ops",
    ],
)
