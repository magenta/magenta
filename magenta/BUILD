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
    default_visibility = ["//visibility:private"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

load("@protobuf//:protobuf.bzl", "cc_proto_library")
load("@protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "music_py_pb2",
    srcs = ["protobuf/music.proto"],
    srcs_version = "PY2AND3",
    protoc = "@protobuf//:protoc",
    default_runtime = "@protobuf//:protobuf_python",
)

cc_proto_library(
   name = "music_proto",
   srcs = ["protobuf/music.proto"],
   protoc = "@protobuf//:protoc",
   default_runtime = "@protobuf//:protobuf",
)

py_library(
    name = "midi_io",
    srcs = ["lib/midi_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":music_py_pb2",
        "@pretty_midi//:pretty_midi",
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
    ],
)

py_library(
    name = "note_sequence_io",
    srcs = ["lib/note_sequence_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":music_py_pb2",
    ],
)

py_test(
    name = "note_sequence_io_test",
    srcs = ["lib/note_sequence_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":note_sequence_io",
    ],
)

py_binary(
    name = "convert_midi_dir_to_note_sequences",
    srcs = ["scripts/convert_midi_dir_to_note_sequences.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":midi_io",
        ":note_sequence_io",
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
   ],
)