# Description:
# Tools and models for using TensorFlow with music and art.

package(
    default_visibility = ["//visibility:private"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

load(
    "magenta",
    "tf_proto_library",
    "tf_proto_library_py",
)

tf_proto_library(
    name = "music_proto",
    srcs = ["protobuf/music.proto"],
)

tf_proto_library_py(
    name = "music_py_pb3",
    srcs = ["protobuf/music.proto"],
)