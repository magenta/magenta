package(default_visibility = ["//magenta:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "export_kmeans_centroids",
    srcs = ["export_kmeans_centroids.py"],
    deps = [
        "//magenta/models/omr/glyphs:corpus",
        "//magenta/models/omr/glyphs:knn_model",
        # absl dep
        # tensorflow dep
    ],
)

py_binary(
    name = "gen_structure_test_case",
    srcs = ["gen_structure_test_case.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/omr:engine",
        # absl dep
    ],
)
