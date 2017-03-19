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
# Tools and models for using TensorFlow with music and art.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "rl_tuner",
    srcs = ["rl_tuner.py"],
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":note_rnn_loader",
        ":rl_tuner_ops",
        ":rl_tuner_eval_metrics",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "rl_tuner_train",
    srcs = ["rl_tuner_train.py"],
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":rl_tuner",
        ":rl_tuner_ops",
        "//magenta/common:tf_lib",
    ],
)

py_library(
    name = "note_rnn_loader",
    srcs = ["note_rnn_loader.py"],
    deps = [
        ":rl_tuner_ops",
        "//magenta/models/shared:events_rnn_graph",
        "//magenta/music:melodies_lib",
        "//magenta/music:midi_io",
        "//magenta/music:sequences_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "rl_tuner_eval_metrics",
    srcs = ["rl_tuner_eval_metrics.py"],
    deps = [
        ":rl_tuner_ops",
        # tensorflow dep
    ],
)

py_library(
    name = "rl_tuner_ops",
    srcs = ["rl_tuner_ops.py"],
    deps = [
        "//magenta/common:tf_lib",
        # tensorflow dep
    ],
)

py_test(
    name = "rl_tuner_test",
    size = "medium",
    srcs = ["rl_tuner_test.py"],
    deps = [
        ":note_rnn_loader",
        ":rl_tuner",
        ":rl_tuner_ops",
    ],
)
