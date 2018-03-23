# Description:
# Glyph classification for OMR.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "glyphs",
    deps = [
        ":base",
        ":corpus",
        ":geometry",
        ":glyph_types",
        ":knn_model",
        ":neural",
        ":note_dots",
        ":saved_classifier",
    ],
)

py_library(
    name = "corpus",
    srcs = ["corpus.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # enum34 dep
        # numpy dep
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "convolutional_test",
    srcs = ["convolutional_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":convolutional",
        ":testing",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # numpy dep
        # pandas dep
        # tensorflow dep
    ],
)

py_library(
    name = "knn_model",
    srcs = ["knn_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "glyph_types",
    srcs = ["glyph_types.py"],
    srcs_version = "PY2AND3",
    deps = ["//magenta/models/omr/protobuf:proto_py_pb2"],
)

py_library(
    name = "neural",
    srcs = ["neural.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "neural_test",
    size = "small",
    srcs = ["neural_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":neural",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "saved_classifier",
    srcs = ["saved_classifier.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convolutional",
        "//magenta/models/omr/staves:staffline_extractor",
        "//magenta/models/omr/util:patches",
        # tensorflow dep
    ],
)

py_test(
    name = "saved_classifier_test",
    srcs = ["saved_classifier_test.py"],
    data = ["//magenta/models/omr/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_classifier",
        "//magenta/models/omr:image",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/structure",
        # backports.tempfile dep
        # tensorflow dep
    ],
)

py_library(
    name = "saved_classifier_fn",
    srcs = ["saved_classifier_fn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_classifier",
        # tensorflow dep
    ],
)

py_library(
    name = "note_dots",
    srcs = ["note_dots.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":geometry",
        ":glyph_types",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/structure:components",
        # numpy dep
    ],
)

py_library(
    name = "repeated",
    srcs = ["repeated.py"],
    srcs_version = "PY2AND3",
    deps = ["//magenta/models/omr/glyphs:glyph_types"],
)

py_library(
    name = "testing",
    testonly = True,
    srcs = ["testing.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convolutional",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # numpy dep
        # tensorflow dep
    ],
)
