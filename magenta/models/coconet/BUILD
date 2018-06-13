# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# Tools and models for using TensorFlow with music and art.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "lib_data",
    srcs = ["lib_data.py"],
    data = [
        "testdata/TestData.npz",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":lib_mask",
        ":lib_pianoroll",
        ":lib_util",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "lib_mask",
    srcs = ["lib_mask.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lib_util",
        # numpy dep
    ],
)

py_library(
    name = "lib_util",
    srcs = ["lib_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "lib_tfutil",
    srcs = ["lib_tfutil.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "lib_pianoroll",
    srcs = ["lib_pianoroll.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep
        "@pretty_midi",
        # tensorflow dep
    ],
)

py_library(
    name = "lib_graph",
    srcs = ["lib_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lib_hparams",
        ":lib_tfutil",
        # tensorflow dep
    ],
)

py_library(
    name = "lib_hparams",
    srcs = ["lib_hparams.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lib_util",
        # tensorflow dep
        # yaml dep
    ],
)

py_library(
    name = "lib_logging",
    srcs = ["lib_logging.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lib_util",
        # numpy dep
    ],
)

py_library(
    name = "lib_sampling",
    srcs = ["lib_sampling.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lib_data",
        ":lib_logging",
        ":lib_mask",
        ":lib_tfutil",
        ":lib_util",
        # numpy dep
    ],
)

py_library(
    name = "lib_evaluation",
    srcs = ["lib_evaluation.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lib_tfutil",
        ":lib_util",
        # numpy dep
        # scipy dep
    ],
)

py_library(
    name = "lib_saved_model",
    srcs = ["lib_saved_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal models
    ],
    deps = [
        # tensorflow dep
    ],
)

py_binary(
    name = "export_saved_model",
    srcs = ["export_saved_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
    ],
    deps = [
        ":lib_graph",
        ":lib_saved_model",
        ":lib_tfsampling",
        # tensorflow dep
    ],
)

py_binary(
    name = "coconet_train",
    srcs = ["coconet_train.py"],
    data = [
        "testdata/TestData.npz",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        # internal models
    ],
    deps = [
        ":lib_data",
        ":lib_graph",
        ":lib_hparams",
        ":lib_util",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "coconet_evaluate",
    srcs = ["coconet_evaluate.py"],
    data = [
        "samples/generated_result.npy",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        # internal models
    ],
    deps = [
        ":lib_data",
        ":lib_evaluation",
        ":lib_graph",
        ":lib_util",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "coconet_sample",
    srcs = ["coconet_sample.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
    ],
    deps = [
        ":lib_graph",
        ":lib_logging",
        ":lib_mask",
        ":lib_pianoroll",
        ":lib_sampling",
        ":lib_tfsampling",
        ":lib_util",
        # numpy dep
        "@pretty_midi",
        # tensorflow dep
    ],
)

py_binary(
    name = "lib_tfsampling",
    srcs = ["lib_tfsampling.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        # internal models
    ],
    deps = [
        ":lib_graph",
        ":lib_hparams",
        ":lib_pianoroll",
        ":lib_util",
        # numpy dep
        # tensorflow dep
    ],
)
