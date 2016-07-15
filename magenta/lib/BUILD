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

py_library(
    name = "melodies_lib",
    srcs = ["melodies_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequence_example_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ]
)

py_test(
    name = "melodies_lib_test",
    srcs = ["melodies_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        ":sequence_example_lib",
        "@protobuf//:protobuf_python",
        # tensorflow dep
    ],
)

py_library(
    name = "midi_io",
    srcs = ["midi_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/protobuf:music_py_pb2",
        "@pretty_midi//:pretty_midi",
        # tensorflow dep
    ],
)

py_test(
    name = "midi_io_test",
    srcs = ["midi_io_test.py"],
    data = ["//magenta/testdata:testdata"],
    srcs_version = "PY2AND3",
    deps = [
        ":midi_io",
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
    name = "sequence_example_lib",
    srcs = ["sequence_example_lib.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "sequence_to_melodies",
    srcs = ["sequence_to_melodies.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        "//magenta/protobuf:music_py_pb2",
        ":note_sequence_io",
        # tensorflow dep
    ],
)

py_test(
    name = "sequence_to_melodies_test",
    srcs = ["sequence_to_melodies_test.py"],
    data = ["//magenta/testdata:testdata"],
    srcs_version = "PY2AND3",
    deps = [
        ":melodies_lib",
        ":sequence_to_melodies",
        # tensorflow dep
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
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "testing_lib",
    srcs = ["testing_lib.py"],
    deps = [
        ":sequences_lib",
        "@protobuf//:protobuf_python",
    ]
)

py_library(
    name = "tf_lib",
    srcs = ["tf_lib.py"],
    srcs_version = "PY2AND3",
)