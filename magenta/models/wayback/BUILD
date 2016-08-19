package(
    default_visibility = [
        "//magenta:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "//magenta/models/wayback/lib:evaluation",
        "//magenta/models/wayback/lib:hyperparameters",
        "//magenta/models/wayback/lib:models",
        "//magenta/models/wayback/lib:namespace",
        "//magenta/models/wayback/lib:training",
        "//magenta/models/wayback/lib:wavefile",
        # tensorflow dep
    ],
)

py_binary(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        "//magenta/models/wayback/lib:hyperparameters",
        "//magenta/models/wayback/lib:models",
        "//magenta/models/wayback/lib:sampling",
        "//magenta/models/wayback/lib:util",
        "//magenta/models/wayback/lib:wavefile",
        # tensorflow dep
    ],
)
