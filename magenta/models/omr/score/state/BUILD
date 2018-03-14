# Description:
# The music score state. Holds information such as the key signature and current time in the score,
# which is used for reading notes from the score.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "state",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":measure",
        ":staff",
    ],
)

py_library(
    name = "measure",
    srcs = ["measure.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/score/elements:key_signature",
        "//magenta/protobuf:music_py_pb2",
        # enum34 dep
    ],
)

py_library(
    name = "staff",
    srcs = ["staff.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":measure",
        "//magenta/models/omr/score/elements:clef",
    ],
)
