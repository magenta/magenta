py_binary(
    name = "rl_tuner_train",
    srcs = ["rl_tuner_train.py"],
    deps = [
        ":rl_tuner",
        ":rl_tuner_ops",
        "//magenta/common:tf_lib",
    ],
)

py_binary(
    name = "rl_tuner",
    srcs = ["rl_tuner.py"],
    deps = [
        ":note_rnn_loader",
        ":rl_tuner_ops",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "note_rnn_loader",
    srcs = ["note_rnn_loader.py"],
    deps = [
        ":rl_tuner_ops",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        "//magenta/music:sequences_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "unpack_ckpt_from_bundle",
    srcs = ["unpack_ckpt_from_bundle.py"],
    deps = [
        ":rl_tuner_ops",
        "//magenta/music:sequence_generator_bundle",
    ],
)

py_binary(
    name = "rl_tuner_ops",
    srcs = ["rl_tuner_ops.py"],
    deps = [
        "//magenta/common:tf_lib",
        # tensorflow dep
    ],
)

py_test(
    name = "rl_tuner_test",
    srcs = ["rl_tuner_test.py"],
    deps = [
        ":rl_tuner",
        ":rl_tuner_ops",
    ],
)
