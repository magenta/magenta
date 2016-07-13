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
    name = "sequences_lib",
    srcs = ["lib/sequences_lib.py"],
)

py_library(
    name = "sequence_example_lib",
    srcs = ["lib/sequence_example_lib.py"],
    deps = [
        "//magenta:melodies_lib",
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipelines/pipeline.py"],
    deps = [
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_library(
    name = "pipeline_unit",
    srcs = ["pipelines/pipeline_unit.py"],
)

py_library(
    name = "pipeline_units_common",
    srcs = ["pipelines/pipeline_units_common.py"],
    deps = [
        ":melodies_lib",
        ":pipeline_unit",
        ":sequences_lib",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_library(
    name = "testing_lib",
    srcs = ["lib/testing_lib.py"],
    deps = [
        ":sequences_lib",
        "@protobuf//:protobuf_python",
    ]
)

py_test(
    name = "melodies_lib_test",
    srcs = ["lib/melodies_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        ":sequences_lib",
        ":testing_lib",
        # tensorflow dep
    ],
)

py_test(
    name = "sequences_lib_test",
    srcs = ["lib/sequences_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequences_lib",
        ":testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "sequence_example_lib_test",
    srcs = ["lib/sequence_example_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        ":sequence_example_lib",
        ":testing_lib",
        # tensorflow dep
    ],
)

py_test(
    name = "pipeline_test",
    srcs = ["pipelines/pipeline_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pipeline",
        # tensorflow dep
    ],
)

py_test(
    name = "pipeline_units_common_test",
    srcs = ["pipelines/pipeline_units_common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        ":pipeline_unit",
        ":pipeline_units_common",
        ":sequences_lib",
        ":testing_lib",
        # tensorflow dep
    ],
)
