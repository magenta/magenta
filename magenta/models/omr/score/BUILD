# Description:
# Score reading for OMR.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "score",
    deps = [
        ":measures",
        ":reader",
    ],
)

py_library(
    name = "measures",
    srcs = ["measures.py"],
    deps = ["//magenta/models/omr/protobuf:proto_py_pb2"],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    deps = [
        ":measures",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/score/elements:clef",
        "//magenta/models/omr/score/state",
    ],
)

py_test(
    name = "reader_test",
    srcs = ["reader_test.py"],
    deps = [
        ":reader",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/conversions",
        "//magenta/protobuf:music_py_pb2",
        # absl/testing dep
        # librosa dep
    ],
)
