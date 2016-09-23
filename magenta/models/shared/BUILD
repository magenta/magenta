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
# Code shared by models.

package(
    default_visibility = [
        "//magenta/models:__subpackages__",
        # internal dev models
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "melody_rnn_create_dataset",
    srcs = ["melody_rnn_create_dataset.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "melody_rnn_create_dataset_test",
    srcs = ["melody_rnn_create_dataset_test.py"],
    deps = [
        ":melody_rnn_create_dataset",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "melody_rnn_generate",
    srcs = ["melody_rnn_generate.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "melody_rnn_sequence_generator",
    srcs = ["melody_rnn_sequence_generator.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "melody_rnn_train",
    srcs = ["melody_rnn_train.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "melody_rnn_graph",
    srcs = ["melody_rnn_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "melody_rnn_graph_test",
    srcs = ["melody_rnn_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melody_rnn_graph",
        "//magenta",
        # tensorflow dep
    ],
)
