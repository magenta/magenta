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

# Description: Shared functionality for models.

package(
    default_visibility = [
        "//magenta/models:__subpackages__",
        # internal models
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "events_rnn_graph",
    srcs = ["events_rnn_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "events_rnn_graph_test",
    srcs = ["events_rnn_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":events_rnn_graph",
        ":events_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "events_rnn_model",
    srcs = ["events_rnn_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":events_rnn_graph",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "events_rnn_train",
    srcs = ["events_rnn_train.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)
