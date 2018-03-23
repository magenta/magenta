# Description:
# Generated training data for OMR.
package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "generation",
    srcs = ["generation.py"],
    data = [":vexflow_generator.js"],
    deps = [
        "//magenta/models/omr:engine",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/staves:staffline_distance",
        "//magenta/models/omr/staves:staffline_extractor",
        "@com_google_protobuf//:protobuf_python",
        # apache-beam dep
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "image_noise",
    srcs = ["image_noise.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)
