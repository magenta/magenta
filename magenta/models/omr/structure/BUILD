# Description:
# Music score structure detection above the staff level.

package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "structure",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":beams",
        ":components",
        ":verticals",
        "//magenta/models/omr/staves",
        "//magenta/models/omr/staves:base",
        "//magenta/models/omr/staves:removal",
        # tensorflow dep
    ],
)

py_test(
    name = "structure_test",
    srcs = ["structure_test.py"],
    data = ["//magenta/models/omr/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        ":structure",
        "//magenta/models/omr:image",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "barlines",
    srcs = ["barlines.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # numpy dep
    ],
)

py_test(
    name = "barlines_test",
    srcs = ["barlines_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":barlines",
        ":beams",
        ":components",
        ":structure",
        ":verticals",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/staves:base",
        # absl/testing dep
        # numpy dep
    ],
)

py_library(
    name = "beams",
    srcs = ["beams.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":components",
        "//magenta/models/omr/vision:morphology",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "beam_processor",
    srcs = ["beam_processor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":components",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/glyphs:glyph_types",
        # numpy dep
    ],
)

py_library(
    name = "components",
    srcs = ["components.py"],
    srcs_version = "PY2AND3",
    deps = [
        # enum34 dep
        # tensorflow dep
    ],
)

py_test(
    name = "components_test",
    srcs = ["components_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":components",
        # tensorflow dep
    ],
)

py_library(
    name = "section_barlines",
    srcs = ["section_barlines.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":barlines",
        ":components",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        # numpy dep
    ],
)

py_library(
    name = "stems",
    srcs = ["stems.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/glyphs:glyph_types",
        # numpy dep
    ],
)

py_test(
    name = "stems_test",
    srcs = ["stems_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":beams",
        ":components",
        ":stems",
        ":structure",
        ":verticals",
        "//magenta/models/omr/protobuf:proto_py_pb2",
        "//magenta/models/omr/staves:base",
        # absl/testing dep
        # numpy dep
    ],
)

py_library(
    name = "verticals",
    srcs = ["verticals.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr/util:functional_ops",
        "//magenta/models/omr/util:memoize",
        "//magenta/models/omr/util:segments",
        "//magenta/models/omr/vision:images",
        "//magenta/models/omr/vision:morphology",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "verticals_test",
    srcs = ["verticals_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":verticals",
        "//magenta/models/omr/staves:testing",
        # numpy dep
        # tensorflow dep
    ],
)
