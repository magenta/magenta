# Research code for "latent transfer"

# licenses(["notice"])  # Apache 2.0

py_library(
    name = "configs",
    srcs = glob([
        "configs/*.py",
        "configs/*/*.py",
        "configs/*/*/*.py",
        "configs/*/*/*/*.py",
    ]),
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":nn",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":local_mnist",
        # PIL dep
        # numpy dep
        # scipy dep
        # sonnet dep
        # tensorflow dep
        # tqdm dep
    ],
)

py_library(
    name = "common_joint",
    srcs = ["common_joint.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":local_mnist",
        # numpy dep
        # scipy dep
        # sonnet dep
        # tensorflow dep
        # tqdm dep
    ],
)

py_binary(
    name = "encode_dataspace",
    srcs = ["encode_dataspace.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":common",
        ":configs",
        ":model_dataspace",
    ],
)

py_library(
    name = "local_mnist",
    srcs = ["local_mnist.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "nn",
    srcs = ["nn.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        # numpy dep
        # sonnet dep
        # tensorflow dep
    ],
)

py_library(
    name = "model_dataspace",
    srcs = ["model_dataspace.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":nn",
        # numpy dep
        # sonnet dep
        # tensorflow dep
    ],
)

py_library(
    name = "model_joint",
    srcs = ["model_joint.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":nn",
        # numpy dep
        # sonnet dep
        # tensorflow dep
    ],
)

py_binary(
    name = "sample_dataspace",
    srcs = ["sample_dataspace.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":common",
        ":configs",
        ":model_dataspace",
    ],
)

py_binary(
    name = "sample_wavegan",
    srcs = ["sample_wavegan.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":common",
        ":configs",
    ],
)

py_binary(
    name = "train_dataspace",
    srcs = ["train_dataspace.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":common",
        ":configs",
        ":model_dataspace",
    ],
)

py_binary(
    name = "train_dataspace_classifier",
    srcs = ["train_dataspace_classifier.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":common",
        ":configs",
        ":model_dataspace",
    ],
)

py_binary(
    name = "train_joint",
    srcs = ["train_joint.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":common",
        ":common_joint",
        ":configs",
        ":model_dataspace",
    ],
)
