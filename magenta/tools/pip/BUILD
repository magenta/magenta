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
        "//magenta/interfaces/midi:midi_clock",
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

        # Music VAE  Model and Scripts
        "//magenta/models/music_vae",
        "//magenta/models/music_vae:music_vae_train",

        # Performance RNN Model and Scripts
        "//magenta/models/performance_rnn",
        "//magenta/models/performance_rnn:performance_rnn_create_dataset",
        "//magenta/models/performance_rnn:performance_rnn_generate",
        "//magenta/models/performance_rnn:performance_rnn_train",

        # Pianoroll RNN-NADE Model and Scripts
        "//magenta/models/pianoroll_rnn_nade",
        "//magenta/models/pianoroll_rnn_nade:pianoroll_rnn_nade_create_dataset",
        "//magenta/models/pianoroll_rnn_nade:pianoroll_rnn_nade_generate",
        "//magenta/models/pianoroll_rnn_nade:pianoroll_rnn_nade_train",

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

        # Arbitrary Image Stylization Scripts
        "//magenta/models/arbitrary_image_stylization:arbitrary_image_stylization_evaluate",
        "//magenta/models/arbitrary_image_stylization:arbitrary_image_stylization_train",
        "//magenta/models/arbitrary_image_stylization:arbitrary_image_stylization_with_weights",

        # SketchRNN Model and Scripts
        "//magenta/models/sketch_rnn",
        "//magenta/models/sketch_rnn:sketch_rnn_train",

        # NSynth
        "//magenta/models/nsynth/wavenet:nsynth_generate",
        "//magenta/models/nsynth/wavenet:nsynth_save_embeddings",

        # Onsets and Frames Transcription
        "//magenta/models/onsets_frames_transcription",
        "//magenta/models/onsets_frames_transcription:onsets_frames_transcription_create_dataset",
        "//magenta/models/onsets_frames_transcription:onsets_frames_transcription_infer",
        "//magenta/models/onsets_frames_transcription:onsets_frames_transcription_train",
        "//magenta/models/onsets_frames_transcription:onsets_frames_transcription_transcribe",
    ],
)
