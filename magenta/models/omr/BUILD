# Description:
# Optical music recognition library for Magenta.

# The OMR engine. Entry point for running OMR.
package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "engine",
    srcs = ["engine.py"],
    data = ["//magenta/models/omr/data/knn_saved_model_20170514"],
    srcs_version = "PY2AND3",
    deps = [
        ":image",
        ":page_processors",
        ":score_processors",
        "//magenta/models/omr/conversions",
        "//magenta/models/omr/glyphs:saved_classifier_fn",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/staves:base",
        "//magenta/models/omr/structure",
        "//magenta/models/omr/structure:beams",
        "//magenta/models/omr/structure:components",
        "//magenta/models/omr/structure:verticals",
        # numpy dep
        # tensorflow dep
    ],
)

# The omr CLI for running locally on a single score.
py_binary(
    name = "omr",
    srcs = ["omr.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        "//magenta/models/omr/conversions",
        "//magenta/models/omr/glyphs:saved_classifier_fn",
        "@com_google_protobuf//:protobuf_python",
        # absl dep
    ],
)

py_test(
    name = "omr_endtoend_test",
    size = "medium",
    srcs = ["omr_endtoend_test.py"],
    data = ["//magenta/models/omr/testdata:images"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        "//magenta/models/omr/conversions",
        "//magenta/protobuf:music_py_pb2",
        # pillow dep
        # absl/testing dep
        # librosa dep
        # lxml dep
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "omr_regression_test",
    size = "medium",
    srcs = ["omr_regression_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    # Requires IMSLP page images to be acquired manually.
    tags = ["manual"],
    deps = [
        ":engine",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/score:reader",
        # absl/testing dep
    ],
)

py_library(
    name = "image",
    srcs = ["image.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "page_processors",
    srcs = ["page_processors.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/glyphs:glyph_types",
        "//magenta/models/omr/glyphs:note_dots",
        "//magenta/models/omr/glyphs:repeated",
        "//magenta/models/omr/staves:staff_processor",
        "//magenta/models/omr/structure:barlines",
        "//magenta/models/omr/structure:beam_processor",
        "//magenta/models/omr/structure:section_barlines",
        "//magenta/models/omr/structure:stems",
    ],
)

py_library(
    name = "score_processors",
    srcs = ["score_processors.py"],
    srcs_version = "PY2AND3",
    deps = ["//magenta/models/omr/score:reader"],
)
