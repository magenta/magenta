# Description:
#   Contains TF code for training wavenet autoencoder models

package(
    default_visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "data",
    srcs = [
        "nsynth_release_sstable.py",
    ],
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "masked",
    srcs = ["masked.py"],
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "config_library",
    deps = [
        ":data",
        ":masked",
        # tensorflow dep
    ],
)

py_library(
    name = "configs",
    srcs = [
        "h512_bo16.py",
    ],
    deps = [
        ":config_library",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        ":configs",
        ":utils",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "save_embeddings",
    srcs = [
        "save_embeddings.py",
    ],
    deps = [
        ":configs",
        ":data",
        ":utils",
        # numpy dep
        # tensorflow dep
    ],
)
