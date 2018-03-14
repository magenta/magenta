# Description:
# Unsupervised learning pipeline for OMR glyph classification.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "staffline_patches_dofn",
    srcs = ["staffline_patches_dofn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr:image",
        "//magenta/models/omr/staves",
        "//magenta/models/omr/staves:removal",
        "//magenta/models/omr/staves:staffline_extractor",
        "//magenta/models/omr/util:patches",
        # apache-beam dep
        # tensorflow dep
    ],
)

py_test(
    name = "staffline_patches_dofn_test",
    size = "medium",
    srcs = ["staffline_patches_dofn_test.py"],
    data = ["//magenta/models/omr/testdata:images"],
    srcs_version = "PY2AND3",
    # Apache Beam
    tags = ["py2only"],
    deps = [
        ":staffline_patches_dofn",
        # absl/testing dep
        # apache-beam dep
        # tensorflow dep
    ],
)

py_binary(
    name = "staffline_patches_kmeans_pipeline",
    srcs = ["staffline_patches_kmeans_pipeline.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":staffline_patches_dofn",
        # absl dep
        # apache-beam dep
        # tensorflow dep
    ],
)

py_test(
    name = "staffline_patches_kmeans_pipeline_test",
    srcs = ["staffline_patches_kmeans_pipeline_test.py"],
    srcs_version = "PY2AND3",
    # Apache Beam
    tags = ["py2only"],
    deps = [
        ":staffline_patches_kmeans_pipeline",
        # absl dep
        # absl/testing dep
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "kmeans_labeler_request_handler",
    srcs = ["kmeans_labeler_request_handler.py"],
    data = ["kmeans_labeler_template.html"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # pillow dep
        # mako dep
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "kmeans_labeler_request_handler_test",
    srcs = ["kmeans_labeler_request_handler_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kmeans_labeler_request_handler",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # absl/testing dep
        # mako dep
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "kmeans_labeler",
    srcs = ["kmeans_labeler.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kmeans_labeler_request_handler",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # pillow dep
        # absl dep
        # mako dep
        # numpy dep
        # tensorflow dep
    ],
)
