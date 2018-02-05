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
    name = "performance_rnn",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":performance_model",
    ],
)

py_library(
    name = "performance_model",
    srcs = ["performance_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        # internal models
    ],
    deps = [
        "//magenta/models/shared:events_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "performance_sequence_generator",
    srcs = ["performance_sequence_generator.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/interfaces:generator_interfaces",
    ],
    deps = [
        ":performance_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "performance_rnn_generate",
    srcs = ["performance_rnn_generate.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":performance_model",
        ":performance_sequence_generator",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "performance_rnn_create_dataset",
    srcs = ["performance_rnn_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:performance_rnn
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":performance_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "performance_rnn_create_dataset_test",
    srcs = ["performance_rnn_create_dataset_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":performance_model",
        ":performance_rnn_create_dataset",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "performance_rnn_train",
    srcs = ["performance_rnn_train.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":performance_model",
        "//magenta/models/shared:events_rnn_graph",
        "//magenta/models/shared:events_rnn_train",
        # tensorflow dep
    ],
)
