# Description:
# General computer vision routines for OMR.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "vision",
    deps = [
        ":hough",
        ":images",
        ":morphology",
    ],
)

py_library(
    name = "images",
    srcs = ["images.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)

py_test(
    name = "images_test",
    srcs = ["images_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":images",
        # tensorflow dep
    ],
)

py_library(
    name = "hough",
    srcs = ["hough.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/util:segments",
        # tensorflow dep
    ],
)

py_test(
    name = "hough_test",
    srcs = ["hough_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hough",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "morphology",
    srcs = ["morphology.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":images",
        # tensorflow dep
    ],
)

py_test(
    name = "morphology_test",
    srcs = ["morphology_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":morphology",
        # numpy dep
        # tensorflow dep
    ],
)
