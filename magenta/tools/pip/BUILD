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

        # Interfaces
        "//magenta/interfaces/midi",

        # Scripts
        "//magenta/interfaces/midi:magenta_midi",
        "//magenta/scripts:convert_dir_to_note_sequences",

        # Drums RNN Model and Scripts
        "//magenta/models/drums_rnn",
        "//magenta/models/drums_rnn:drums_rnn_create_dataset",
        "//magenta/models/drums_rnn:drums_rnn_generate",
        "//magenta/models/drums_rnn:drums_rnn_train",

        # Improv RNN Model and Scripts
        "//magenta/models/improv_rnn",
        "//magenta/models/improv_rnn:improv_rnn_create_dataset",
        "//magenta/models/improv_rnn:improv_rnn_generate",
        "//magenta/models/improv_rnn:improv_rnn_train",

        # Melody RNN Model and Scripts
        "//magenta/models/melody_rnn",
        "//magenta/models/melody_rnn:melody_rnn_create_dataset",
        "//magenta/models/melody_rnn:melody_rnn_generate",
        "//magenta/models/melody_rnn:melody_rnn_train",

        # Polyphony RNN Model and Scripts
        "//magenta/models/polyphony_rnn",
        "//magenta/models/polyphony_rnn:polyphony_rnn_create_dataset",
        "//magenta/models/polyphony_rnn:polyphony_rnn_generate",
        "//magenta/models/polyphony_rnn:polyphony_rnn_train",

        # RL Tuner Model and Scripts
        "//magenta/models/rl_tuner",
        "//magenta/models/rl_tuner:rl_tuner_train",

        # Image Stylization Scripts
        "//magenta/models/image_stylization:image_stylization_create_dataset",
        "//magenta/models/image_stylization:image_stylization_evaluate",
        "//magenta/models/image_stylization:image_stylization_finetune",
        "//magenta/models/image_stylization:image_stylization_train",
        "//magenta/models/image_stylization:image_stylization_transform",
    ],
)
