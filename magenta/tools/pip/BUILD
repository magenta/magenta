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

licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",

        # Libraries
        "//magenta",

        # Scripts
        "//magenta/interfaces/midi:magenta_midi",
        "//magenta/models/attention_rnn:attention_rnn_create_dataset",
        "//magenta/models/attention_rnn:attention_rnn_generate",
        "//magenta/models/attention_rnn:attention_rnn_train",
        "//magenta/models/basic_rnn:basic_rnn_create_dataset",
        "//magenta/models/basic_rnn:basic_rnn_generate",
        "//magenta/models/basic_rnn:basic_rnn_train",
        "//magenta/models/lookback_rnn:lookback_rnn_create_dataset",
        "//magenta/models/lookback_rnn:lookback_rnn_generate",
        "//magenta/models/lookback_rnn:lookback_rnn_train",
        "//magenta/scripts:convert_midi_dir_to_note_sequences",
    ],
)
