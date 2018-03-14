# Description:
# Protobufs defining the OMR music page model.

# The OMR engine. Entry point for running OMR.
package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "proto_py_pb2",
    srcs = glob(["*.proto"]),
    deps = ["//magenta/protobuf:music_py_pb2"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY2AND3",
)

cc_proto_library(
    name = "proto",
    srcs = glob(["*.proto"]),
    deps = ["//magenta/protobuf:music_proto"],
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)
