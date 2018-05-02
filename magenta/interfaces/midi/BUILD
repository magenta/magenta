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
# MIDI interface for Magenta generators.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "midi",
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
        # internal notebook binary
    ],
    deps = [
        ":midi_hub",
        ":midi_interaction",
    ],
)

py_library(
    name = "midi_hub",
    srcs = ["midi_hub.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        "@mido",
        # tensorflow dep
    ],
)

py_test(
    name = "midi_hub_test",
    srcs = ["midi_hub_test.py"],
    srcs_version = "PY2AND3",
    # TODO(fjord): reenable once test is not flaky
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":midi_hub",
        "//magenta",
        "@mido",
        # tensorflow dep
    ],
)

py_library(
    name = "midi_interaction",
    srcs = ["midi_interaction.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":midi_hub",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "magenta_midi",
    srcs = ["magenta_midi.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":midi_hub",
        ":midi_interaction",
        "//magenta",
        "//magenta/models/drums_rnn:drums_rnn_sequence_generator",
        "//magenta/models/melody_rnn:melody_rnn_sequence_generator",
        "//magenta/models/performance_rnn:performance_sequence_generator",
        "//magenta/models/pianoroll_rnn_nade:pianoroll_rnn_nade_sequence_generator",
        "//magenta/models/polyphony_rnn:polyphony_sequence_generator",
    ],
)

py_binary(
    name = "midi_clock",
    srcs = ["midi_clock.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":midi_hub",
    ],
)
