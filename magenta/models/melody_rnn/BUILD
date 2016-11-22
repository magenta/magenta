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

# Description: An RNN model for generating melodies.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "melody_rnn",
    srcs = ["__init__.py"],
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":melody_rnn_model",
    ],
)

py_library(
    name = "melody_rnn_config_flags",
    srcs = ["melody_rnn_config_flags.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:melody_rnn
    ],
    deps = [
        ":melody_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "melody_rnn_model",
    srcs = ["melody_rnn_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        "//magenta/models/shared:events_rnn_model",
    ],
)

py_library(
    name = "melody_rnn_sequence_generator",
    srcs = ["melody_rnn_sequence_generator.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/interfaces:generator_interfaces"],
    deps = [
        ":melody_rnn_model",
        "//magenta",
    ],
)

py_binary(
    name = "melody_rnn_create_dataset",
    srcs = ["melody_rnn_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:melody_rnn
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":melody_rnn_config_flags",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "melody_rnn_create_dataset_test",
    srcs = ["melody_rnn_create_dataset_test.py"],
    deps = [
        ":melody_rnn_create_dataset",
        ":melody_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn_train",
    srcs = ["melody_rnn_train.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":melody_rnn_config_flags",
        "//magenta/models/shared:events_rnn_graph",
        "//magenta/models/shared:events_rnn_train",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn_generate",
    srcs = ["melody_rnn_generate.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":melody_rnn_config_flags",
        ":melody_rnn_model",
        ":melody_rnn_sequence_generator",
        "//magenta",
        # tensorflow dep
    ],
)
