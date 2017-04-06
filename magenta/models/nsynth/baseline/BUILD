# Description:
#   Contains TF code for training baseline specgram models

package(
    default_visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
)


py_library(
    name = "ae_configs",
    srcs = glob(["models/ae_configs/*.py"]),
    deps = [
        "//magenta/common:tf_lib",
        "//magenta/models/nsynth:utils",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "ae",
    srcs = ["models/ae.py"],
    deps = [
        ":ae_configs",
        "//magenta/common:tf_lib",
        "//magenta/models/nsynth:utils",
        # tensorflow dep
    ],
)

py_library(
    name = "library",
    deps = [
        ":ae",
        "//magenta/models/nsynth:reader",
        "//magenta/models/nsynth:utils",
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
    name = "save_embeddings",
    srcs = ["save_embeddings.py"],
    deps = [
        ":library",
    ],
)

