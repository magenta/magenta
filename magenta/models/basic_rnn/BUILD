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
# Basic RNN model.

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "basic_rnn_create_dataset",
    srcs = ["basic_rnn_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal pipeline runner
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":basic_rnn_encoder_decoder",
        "//magenta/models/shared:melody_rnn_create_dataset",
        # tensorflow dep
    ],
)

py_library(
    name = "basic_rnn_encoder_decoder",
    srcs = ["basic_rnn_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal dev models
    ],
    deps = [
        "//magenta",
    ],
)

py_test(
    name = "basic_rnn_encoder_decoder_test",
    srcs = ["basic_rnn_encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_rnn_encoder_decoder",
        "//magenta/music:melodies_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "basic_rnn_generator",
    srcs = ["basic_rnn_generator.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/interfaces:generator_interfaces"],
    deps = [
        ":basic_rnn_encoder_decoder",
        ":basic_rnn_graph",
        "//magenta/models/shared:melody_rnn_sequence_generator",
        "//magenta/protobuf:generator_py_pb2",
    ],
)

py_binary(
    name = "basic_rnn_generate",
    srcs = ["basic_rnn_generate.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":basic_rnn_generator",
        "//magenta/models/shared:melody_rnn_generate",
        # tensorflow dep
    ],
)

py_library(
    name = "basic_rnn_graph",
    srcs = ["basic_rnn_graph.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal dev models
    ],
    deps = [
        "//magenta/models/shared:melody_rnn_graph",
    ],
)

py_test(
    name = "basic_rnn_graph_test",
    srcs = ["basic_rnn_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_rnn_graph",
        "//magenta/music:melodies_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "basic_rnn_train",
    srcs = ["basic_rnn_train.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":basic_rnn_encoder_decoder",
        ":basic_rnn_graph",
        "//magenta/models/shared:melody_rnn_train",
        # tensorflow dep
    ],
)
