package(
    default_visibility = [
        "//magenta/models/wayback:__subpackages__",
    ],
)

py_library(
    name = "namespace",
    srcs = ["namespace.py"],
)

py_test(
    name = "namespace_test",
    size = "small",
    srcs = ["namespace_test.py"],
    deps = [
        ":namespace",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        ":namespace",
        # numpy dep
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        # numpy dep
    ],
)

py_library(
    name = "tfutil",
    srcs = ["tfutil.py"],
    deps = [
        ":namespace",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    deps = [
        ":namespace",
        ":util",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "hyperparameters_test",
    size = "small",
    srcs = ["hyperparameters_test.py"],
    deps = [
        ":hyperparameters",
        ":namespace",
    ],
)

py_library(
    name = "wavefile",
    srcs = ["wavefile.py"],
    deps = [
        ":namespace",
        # numpy dep
        # scipy dep
        # tensorflow dep
    ],
)

py_library(
    name = "cells",
    srcs = ["cells.py"],
    deps = [
        ":tfutil",
        ":util",
        # tensorflow dep
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        ":namespace",
        ":tfutil",
        ":util",
        ":cells",
        # tensorflow dep
    ],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    deps = [
        ":namespace",
        ":tfutil",
        ":util",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":namespace",
        ":util",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    deps = [
        ":namespace",
        ":util",
        # numpy dep
        # tensorflow dep
    ],
)
