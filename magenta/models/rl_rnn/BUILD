py_binary(
    name = "melody_q_train",
    srcs = ["melody_q_train.py"],
    deps = [
        ":melody_q",
        ":rl_rnn_ops",
    ],
)

py_binary(
    name = "melody_q",
    srcs = ["melody_q.py"],
    deps = [
        ":melody_rnn",
        ":rl_rnn_ops",
        "//magenta/lib:melodies_lib",
        "//magenta/lib:midi_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn",
    srcs = ["melody_rnn.py"],
    deps = [
        ":melody_rnn_encoder_decoder",
        ":rl_rnn_ops",
        tensor flow ops
        "//magenta/lib:melodies_lib",
        "//magenta/lib:midi_io",
        "//magenta/lib:sequences_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn_encoder_decoder",
    srcs = ["melody_rnn_encoder_decoder.py"],
    deps = [
        "//magenta/lib:melodies_lib",
    ],
)

py_binary(
    name = "rl_rnn_ops",
    srcs = ["rl_rnn_ops.py"],
    deps = [
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
