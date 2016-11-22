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
# Multi-event RNN model.

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "multi_event_rnn_train",
    srcs = ["multi_event_rnn_train.py"],
    deps = [
        ":multi_event_rnn_lib",
        ":multi_event_rnn_graph",
        # tensorflow dep
    ],
)

py_binary(
    name = "multi_event_rnn_generate",
    srcs = ["multi_event_rnn_generate.py"],
    deps = [
        ":multi_event_rnn_lib",
        ":multi_event_rnn_graph",
        # tensorflow dep
    ],
)

py_library(
    name = "multi_event_rnn_graph",
    srcs = ["multi_event_rnn_graph.py"],
    deps = [
        ":multi_event_rnn_lib",
    ],
)

py_library(
    name = "multi_event_rnn_lib",
    srcs = ["multi_event_rnn_lib.py"],
    deps = [
        "//magenta",
        # numpy dep
        # scipy dep
        # tensorflow dep
    ],
)
