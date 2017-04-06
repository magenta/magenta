# Description:
#   Contains TF code for training baseline specgram models

package(
    default_visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # librosa dep
        # numpy dep
        # tensorflow dep
        # scipy dep
    ],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    deps = [
        ":datasets",
        ":utils",
        # numpy dep
        # tensorflow dep
   ],
)

py_library(
    name = "ae_configs",
    srcs = glob(["models/ae_configs/*.py"]),
    deps = [
        ":utils",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "ae",
    srcs = ["models/ae.py"],
    deps = [
        ":ae_configs",
        ":utils",
        # tensorflow dep
    ],
)

py_library(
    name = "pitch_configs",
    srcs = glob(["models/pitch_configs/*.py"]),
    deps = [
        ":utils",
        # tensorflow dep
    ],
    ],
)

py_library(
    name = "pitch",
    srcs = ["models/pitch.py"],
    deps = [
        ":pitch_configs",
        ":utils",
        # tensorflow dep
    ],
)

py_library(
    name = "library",
    deps = [
        ":ae",
        ":datasets",
        ":pitch",
        ":reader",
        ":utils",
        # librosa dep
        # tensorflow dep
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":library",
    ],
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":library",
    ],
)

py_binary(
    name = "interp",
    srcs = ["interp.py"],
    deps = [
        ":library",
    ],
)

py_binary(
    name = "save_z",
    srcs = ["save_z.py"],
    deps = [
        ":library",
    ],
)

py_binary(
    name = "make_test_set",
    srcs = ["make_test_set.py"],
    deps = [
        ":library",
    ],
)
