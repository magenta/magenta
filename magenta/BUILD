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

package(
    default_visibility = ["//magenta:__subpackages__"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0
exports_files(["LICENSE"])

py_library(
    name = "midi_io",
    srcs = ["lib/midi_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/protobuf:music_py_pb2",
        "@pretty_midi//:pretty_midi",
        # tensorflow dep
    ],
)

py_test(
    name = "midi_io_test",
    srcs = ["lib/midi_io_test.py"],
    srcs_version = "PY2AND3",
    data = [
        "testdata/example.mid",
        "testdata/example_complex.mid",
    ],
    deps = [
        ":midi_io",
        # tensorflow dep
    ],
)

py_library(
    name = "note_sequence_io",
    srcs = ["lib/note_sequence_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "note_sequence_io_test",
    srcs = ["lib/note_sequence_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":note_sequence_io",
        # tensorflow dep
    ],
)

py_binary(
    name = "convert_midi_dir_to_note_sequences",
    srcs = ["scripts/convert_midi_dir_to_note_sequences.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":midi_io",
        ":note_sequence_io",
        # tensorflow dep
    ],
)

py_test(
   name = "convert_midi_dir_to_note_sequences_test",
   srcs = ["scripts/convert_midi_dir_to_note_sequences_test.py"],
   srcs_version = "PY2AND3",
   data = [
       "testdata/example.mid",
   ],
   deps = [
       ":convert_midi_dir_to_note_sequences",
        # tensorflow dep
   ],
)

py_library(
    name = "melodies_lib",
    srcs = ["lib/melodies_lib.py"],
    deps = [
        "//magenta/protobuf:music_py_pb2",
    ]
)

py_library(
    name = "sequence_to_melodies",
    srcs = ["lib/sequence_to_melodies.py"],
    deps = [
        ":melodies_lib",
        "//magenta/protobuf:music_py_pb2",
        ":note_sequence_io",
        # tensorflow dep
    ],
)

py_test(
    name = "melodies_lib_test",
    srcs = ["lib/melodies_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        "@protobuf//:protobuf_python",
        # tensorflow dep
    ],
)

py_test(
    name = "sequence_to_melodies_test",
    srcs = ["lib/sequence_to_melodies_test.py"],
    srcs_version = "PY2AND3",
    data = [
        "testdata/notesequences.tfrecord",
    ],
    deps = [
        ":sequence_to_melodies",
        # tensorflow dep
    ],
)

py_test(
    name = "basic_one_hot_encoder_test",
    srcs = ["lib/basic_one_hot_encoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        ":sequence_to_melodies",
        # tensorflow dep
    ],
)
