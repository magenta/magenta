# Description:
# Score elements which encapsulate custom logic for score reading.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "clef",
    srcs = ["clef.py"],
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/music:constants",
        # librosa dep
    ],
)

py_test(
    name = "clef_test",
    srcs = ["clef_test.py"],
    deps = [
        ":clef",
        # absl/testing dep
        # librosa dep
    ],
)

py_library(
    name = "key_signature",
    srcs = ["key_signature.py"],
    deps = [
        ":clef",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/music:constants",
        # librosa dep
    ],
)

py_test(
    name = "key_signature_test",
    srcs = ["key_signature_test.py"],
    deps = [
        ":clef",
        ":key_signature",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # absl/testing dep
    ],
)
