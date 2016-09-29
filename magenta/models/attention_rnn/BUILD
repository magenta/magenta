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
# Attention RNN model.

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "attention_rnn_create_dataset",
    srcs = ["attention_rnn_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal pipeline runner
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":attention_rnn_encoder_decoder",
        "//magenta/models/shared:melody_rnn_create_dataset",
        # tensorflow dep
    ],
)

py_library(
    name = "attention_rnn_encoder_decoder",
    srcs = ["attention_rnn_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal dev models
    ],
    deps = [
        "//magenta",
    ],
)

py_test(
    name = "attention_rnn_encoder_decoder_test",
    srcs = ["attention_rnn_encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":attention_rnn_encoder_decoder",
        "//magenta/music:melodies_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "attention_rnn_generator",
    srcs = ["attention_rnn_generator.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/interfaces:generator_interfaces"],
    deps = [
        ":attention_rnn_encoder_decoder",
        ":attention_rnn_graph",
        "//magenta",
        "//magenta/models/shared:melody_rnn_sequence_generator",
    ],
)

py_binary(
    name = "attention_rnn_generate",
    srcs = ["attention_rnn_generate.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":attention_rnn_generator",
        "//magenta/models/shared:melody_rnn_generate",
        # tensorflow dep
    ],
)

py_library(
    name = "attention_rnn_graph",
    srcs = ["attention_rnn_graph.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal dev models
    ],
    deps = [
        "//magenta",
        "//magenta/models/shared:melody_rnn_graph",
    ],
)

py_test(
    name = "attention_rnn_graph_test",
    srcs = ["attention_rnn_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":attention_rnn_graph",
        "//magenta/music:melodies_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "attention_rnn_train",
    srcs = ["attention_rnn_train.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":attention_rnn_encoder_decoder",
        ":attention_rnn_graph",
        "//magenta/models/shared:melody_rnn_train",
        # tensorflow dep
    ],
)
