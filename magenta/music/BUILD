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
# Libraries for using TensorFlow with music and art.

package(
    default_visibility = ["//magenta:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

# The Magenta public API.
py_library(
    name = "music",
    srcs = ["__init__.py"],
    deps = [
        ":constants",
        ":melodies_lib",
        ":melody_encoder_decoder",
        ":midi_io",
        ":midi_synth",
        ":model",
        ":note_sequence_io",
        ":notebook_utils",
        ":sequence_generator",
        ":sequence_generator_bundle",
        ":sequences_lib",
        ":testing_lib",
    ],
)

py_library(
    name = "chord_symbols_lib",
    srcs = ["chord_symbols_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@music21//:music21",
        # tensorflow dep
    ],
)

py_test(
    name = "chord_symbols_lib_test",
    srcs = ["chord_symbols_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chord_symbols_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "chords_lib",
    srcs = ["chords_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chord_symbols_lib",
        ":constants",
        ":events_lib",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "chords_lib_test",
    srcs = ["chords_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chord_symbols_lib",
        ":chords_lib",
        ":constants",
        ":melodies_lib",
        ":sequences_lib",
        ":testing_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "events_lib",
    srcs = ["events_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        "//magenta/common:sequence_example_lib",
        # numpy dep
    ],
)

py_test(
    name = "events_lib_test",
    srcs = ["events_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":events_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "lead_sheets_lib",
    srcs = ["lead_sheets_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chords_lib",
        ":constants",
        ":events_lib",
        ":melodies_lib",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "lead_sheets_lib_test",
    srcs = ["lead_sheets_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chords_lib",
        ":constants",
        ":lead_sheets_lib",
        ":melodies_lib",
        ":sequences_lib",
        ":testing_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "melodies_lib",
    srcs = ["melodies_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":events_lib",
        ":midi_io",
        ":sequences_lib",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
        # numpy dep
    ],
)

py_test(
    name = "melodies_lib_test",
    srcs = ["melodies_lib_test.py"],
    data = [
        "testdata/melody.mid",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":melodies_lib",
        ":sequences_lib",
        ":testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "melody_encoder_decoder",
    srcs = ["melody_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":events_lib",
        ":melodies_lib",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
        # numpy dep
    ],
)

py_test(
    name = "melody_encoder_decoder_test",
    srcs = ["melody_encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":melodies_lib",
        ":melody_encoder_decoder",
        ":sequences_lib",
        "//magenta/common:sequence_example_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "midi_io",
    srcs = ["midi_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/music:constants",
        "//magenta/protobuf:music_py_pb2",
        "@pretty_midi//:pretty_midi",
        # tensorflow dep
    ],
)

py_library(
    name = "midi_synth",
    srcs = ["midi_synth.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":midi_io",
        # numpy dep
    ],
)

py_test(
    name = "midi_io_test",
    srcs = ["midi_io_test.py"],
    data = ["//magenta/testdata"],
    srcs_version = "PY2AND3",
    deps = [
        ":midi_io",
        "@mido//:mido",
        "@pretty_midi//:pretty_midi",
        # tensorflow dep
    ],
)

py_library(
    name = "note_sequence_io",
    srcs = ["note_sequence_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "note_sequence_io_test",
    srcs = ["note_sequence_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":note_sequence_io",
        # tensorflow dep
    ],
)

py_library(
    name = "notebook_utils",
    srcs = ["notebook_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":midi_synth",
        # IPython dep
    ],
)

py_library(
    name = "sequences_lib",
    srcs = ["sequences_lib.py"],
)

py_test(
    name = "sequences_lib_test",
    srcs = ["sequences_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequences_lib",
        ":testing_lib",
        "//magenta/common:testing_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "sequence_generator_bundle",
    srcs = ["sequence_generator_bundle.py"],
    deps = [
        "@protobuf//:protobuf_python",
        "//magenta/protobuf:generator_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "sequence_generator",
    srcs = ["sequence_generator.py"],
    deps = [
        "//magenta/protobuf:generator_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "sequence_generator_test",
    srcs = ["sequence_generator_test.py"],
    deps = [
        "//magenta/protobuf:generator_py_pb2",
        ":model",
        ":sequence_generator",
        # tensorflow dep
    ],
)

py_library(
    name = "testing_lib",
    srcs = ["testing_lib.py"],
    deps = [
        ":sequences_lib",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_library(
    name = "pretty_music21",
    srcs = ["pretty_music21.py"],
    deps = [
        "@music21//:music21",
    ],
)

py_library(
    name = "music21_to_note_sequence_io",
    srcs = ["music21_to_note_sequence_io.py"],
    deps = [
        ":pretty_music21",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_library(
    name = "music_xml_io",
    srcs = ["music_xml_io.py"],
    deps = [
        ":music21_to_note_sequence_io",
    ],
)

py_test(
    name = "pretty_music21_test",
    size = "medium",
    srcs = ["pretty_music21_test.py"],
    deps = [
        ":pretty_music21",
        "@music21//:music21",
        # tensorflow dep
    ],
)

py_test(
    name = "music21_to_note_sequence_io_test",
    size = "small",
    srcs = ["music21_to_note_sequence_io_test.py"],
    data = [
        "testdata/bach-one_phrase-4_voices.xml",
    ],
    deps = [
        ":music21_to_note_sequence_io",
        # tensorflow dep
    ],
)

py_test(
    name = "music_xml_io_test",
    size = "small",
    srcs = ["music_xml_io_test.py"],
    data = [
        "testdata/bach-one_phrase-4_voices.xml",
    ],
    deps = [
        ":music_xml_io",
        # tensorflow dep
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        # tensorflow dep
    ],
)
