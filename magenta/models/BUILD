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
# Models for using TensorFlow with music and art.

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "attention_rnn_create_dataset",
    srcs = ["attention_rnn/attention_rnn_create_dataset.py"],
    deps = [
        ":attention_rnn_encoder_decoder",
        ":melody_rnn_create_dataset",
        # tensorflow dep
    ],
)

py_binary(
    name = "attention_rnn_encoder_decoder",
    srcs = ["attention_rnn/attention_rnn_encoder_decoder.py"],
    deps = [
        "//magenta:melodies_lib",
    ],
)

py_binary(
    name = "attention_rnn_generate",
    srcs = ["attention_rnn/attention_rnn_generate.py"],
    deps = [
        ":attention_rnn_encoder_decoder",
        ":attention_rnn_graph",
        ":melody_rnn_generate",
        # tensorflow dep
    ],
)

py_binary(
    name = "attention_rnn_graph",
    srcs = ["attention_rnn/attention_rnn_graph.py"],
    deps = [
        "//magenta:sequence_example_lib",
        "//magenta:tf_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "attention_rnn_train",
    srcs = ["attention_rnn/attention_rnn_train.py"],
    deps = [
        ":attention_rnn_encoder_decoder",
        ":attention_rnn_graph",
        ":melody_rnn_train",
        # tensorflow dep
    ],
)

py_binary(
    name = "basic_rnn_create_dataset",
    srcs = ["basic_rnn/basic_rnn_create_dataset.py"],
    deps = [
        ":basic_rnn_encoder_decoder",
        ":melody_rnn_create_dataset",
        # tensorflow dep
    ],
)

py_binary(
    name = "basic_rnn_encoder_decoder",
    srcs = ["basic_rnn/basic_rnn_encoder_decoder.py"],
    deps = [
        "//magenta:melodies_lib",
    ],
)

py_binary(
    name = "basic_rnn_generate",
    srcs = ["basic_rnn/basic_rnn_generate.py"],
    deps = [
        ":basic_rnn_encoder_decoder",
        ":basic_rnn_graph",
        ":melody_rnn_generate",
        # tensorflow dep
    ],
)

py_binary(
    name = "basic_rnn_graph",
    srcs = ["basic_rnn/basic_rnn_graph.py"],
    deps = [
        "//magenta:sequence_example_lib",
        "//magenta:tf_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "basic_rnn_train",
    srcs = ["basic_rnn/basic_rnn_train.py"],
    deps = [
        ":basic_rnn_encoder_decoder",
        ":basic_rnn_graph",
        ":melody_rnn_train",
        # tensorflow dep
    ],
)

py_binary(
    name = "lookback_rnn_create_dataset",
    srcs = ["lookback_rnn/lookback_rnn_create_dataset.py"],
    deps = [
        ":lookback_rnn_encoder_decoder",
        ":melody_rnn_create_dataset",
        # tensorflow dep
    ],
)

py_binary(
    name = "lookback_rnn_encoder_decoder",
    srcs = ["lookback_rnn/lookback_rnn_encoder_decoder.py"],
    deps = [
        "//magenta:melodies_lib",
    ],
)

py_binary(
    name = "lookback_rnn_generate",
    srcs = ["lookback_rnn/lookback_rnn_generate.py"],
    deps = [
        ":lookback_rnn_encoder_decoder",
        ":lookback_rnn_graph",
        ":melody_rnn_generate",
        # tensorflow dep
    ],
)

py_binary(
    name = "lookback_rnn_graph",
    srcs = ["lookback_rnn/lookback_rnn_graph.py"],
    deps = [
        "//magenta:sequence_example_lib",
        "//magenta:tf_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "lookback_rnn_train",
    srcs = ["lookback_rnn/lookback_rnn_train.py"],
    deps = [
        ":lookback_rnn_encoder_decoder",
        ":lookback_rnn_graph",
        ":melody_rnn_train",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn_create_dataset",
    srcs = ["shared/melody_rnn_create_dataset.py"],
    deps = [
        "//magenta:sequence_to_melodies",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn_generate",
    srcs = ["shared/melody_rnn_generate.py"],
    deps = [
        "//magenta:melodies_lib",
        "//magenta:midi_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "melody_rnn_train",
    srcs = ["shared/melody_rnn_train.py"],
    deps = [
        # tensorflow dep
    ],
)

