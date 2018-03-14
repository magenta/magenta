# Description:
# OMR evaluation and music score similarity metrics.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "evaluation",
    deps = [
        ":evaluator",
        ":musicxml",
    ],
)

py_binary(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":musicxml",
        "//magenta/models/omr:engine",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/conversions",
        "@com_google_protobuf//:protobuf_python",
        # tensorflow dep
    ],
)

py_test(
    name = "evaluator_endtoend_test",
    srcs = ["evaluator_endtoend_test.py"],
    data = [
        "//magenta/models/omr/testdata:images",
        "//magenta/models/omr/testdata:musicxml",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":evaluator",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "musicxml",
    srcs = ["musicxml.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/music:constants",
        # lxml dep
        # pandas dep
    ],
)

py_test(
    name = "musicxml_test",
    srcs = ["musicxml_test.py"],
    data = ["//magenta/models/omr/testdata:musicxml"],
    srcs_version = "PY2AND3",
    deps = [
        ":musicxml",
        # absl/testing dep
        # lxml dep
        # pandas dep
        # tensorflow dep
    ],
)
