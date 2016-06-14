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

py_binary(
    name = "basic_rnn_train",
    srcs = ["basic_rnn/basic_rnn_train.py"],
    deps = [
        ":basic_rnn_ops",
    ],
)

py_binary(
    name = "basic_rnn_generate",
    srcs = ["basic_rnn/basic_rnn_generate.py"],
    deps = [
        ":basic_rnn_ops",
        "//magenta:melodies_lib",
        "//magenta:midi_io",
        "//magenta:sequence_to_melodies",
    ],
)

py_binary(
    name = "create_dataset",
    srcs = ["basic_rnn/create_dataset.py"],
    deps = [
        "//magenta:melodies_lib",
        "//magenta:midi_io",
        "//magenta:sequence_to_melodies",
    ],
)

py_library(
    name = "basic_rnn_ops",
    srcs = ["basic_rnn/basic_rnn_ops.py"],
    deps = [
    ],
)

py_test(
    name = "basic_rnn_train_test",
    srcs = ["basic_rnn/basic_rnn_train_test.py"],
    srcs_version = "PY2AND3",
    data = [
        "basic_rnn/testdata/melodies.sample.tfrecord",
    ],
    deps = [
        ":basic_rnn_train",
    ],
)
