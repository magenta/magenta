# Description:
# Format conversions for OMR.
package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "conversions",
    srcs = ["__init__.py"],
    deps = [
        ":musicxml",
        ":notesequence",
    ],
)

py_library(
    name = "musicxml",
    srcs = ["musicxml.py"],
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/score:measures",
        # librosa dep
        # lxml dep
    ],
)

py_test(
    name = "musicxml_test",
    srcs = ["musicxml_test.py"],
    deps = [
        ":musicxml",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/protobuf:music_py_pb2",
        # absl/testing dep
    ],
)

py_library(
    name = "notesequence",
    srcs = ["notesequence.py"],
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/protobuf:music_py_pb2",
    ],
)
