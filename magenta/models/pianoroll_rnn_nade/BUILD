# Copyright 2017 Google Inc. All Rights Reserved.
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

licenses(["notice"])  # Apache 2.0

py_library(
    name = "pianoroll_rnn_nade",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":pianoroll_rnn_nade_model",
    ],
)

py_library(
    name = "pianoroll_rnn_nade_graph",
    srcs = ["pianoroll_rnn_nade_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        "//magenta/models/shared:events_rnn_graph",
        # tensorflow dep
    ],
)

py_library(
    name = "pianoroll_rnn_nade_model",
    srcs = ["pianoroll_rnn_nade_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        # internal model:pianoroll_rnn_nade
    ],
    deps = [
        ":pianoroll_rnn_nade_graph",
        "//magenta",
        "//magenta/models/shared:events_rnn_model",
        # tensorflow dep
    ],
)

py_library(
    name = "pianoroll_rnn_nade_sequence_generator",
    srcs = ["pianoroll_rnn_nade_sequence_generator.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/interfaces:generator_interfaces",
    ],
    deps = [
        ":pianoroll_rnn_nade_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "pianoroll_rnn_nade_generate",
    srcs = ["pianoroll_rnn_nade_generate.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":pianoroll_rnn_nade_model",
        ":pianoroll_rnn_nade_sequence_generator",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "pianoroll_rnn_nade_create_dataset",
    srcs = ["pianoroll_rnn_nade_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:pianoroll_rnn_nade
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":pianoroll_rnn_nade_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "pianoroll_rnn_nade_create_dataset_test",
    srcs = ["pianoroll_rnn_nade_create_dataset_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pianoroll_rnn_nade_create_dataset",
        "//magenta",
        "//magenta/models/shared:events_rnn_model",
        # tensorflow dep
    ],
)

py_binary(
    name = "pianoroll_rnn_nade_train",
    srcs = ["pianoroll_rnn_nade_train.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":pianoroll_rnn_nade_graph",
        ":pianoroll_rnn_nade_model",
        "//magenta/models/shared:events_rnn_train",
        # tensorflow dep
    ],
)
