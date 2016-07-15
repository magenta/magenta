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
    deps = [
        ":basic_rnn_encoder_decoder",
        "//magenta/models/shared:melody_rnn_create_dataset",
        # tensorflow dep
    ],
)

py_binary(
    name = "basic_rnn_encoder_decoder",
    srcs = ["basic_rnn_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/lib:melodies_lib",
    ],
)

py_binary(
    name = "basic_rnn_generate",
    srcs = ["basic_rnn_generate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_rnn_encoder_decoder",
        ":basic_rnn_graph",
        "//magenta/models/shared:melody_rnn_generate",
    ],
)

py_binary(
    name = "basic_rnn_graph",
    srcs = ["basic_rnn_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/lib:sequence_example_lib",
        "//magenta/lib:tf_lib",
        # tensorflow dep
    ],
)

py_binary(
    name = "basic_rnn_train",
    srcs = ["basic_rnn_train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_rnn_encoder_decoder",
        ":basic_rnn_graph",
        "//magenta/models/shared:melody_rnn_train",
        # tensorflow dep
    ],
)
