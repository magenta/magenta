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
# Libraries for musical event sequences.

package(
    default_visibility = ["//magenta:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "chords_lib",
    srcs = ["chords_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":events_lib",
        "//magenta/music:chord_symbols_lib",
        "//magenta/music:constants",
        "//magenta/music:sequences_lib",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "chords_lib_test",
    srcs = ["chords_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chords_lib",
        ":melodies_lib",
        "//magenta/common:testing_lib",
        "//magenta/music:chord_symbols_lib",
        "//magenta/music:constants",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "drums_lib",
    srcs = ["drums_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":events_lib",
        "//magenta/music:constants",
        "//magenta/music:midi_io",
        "//magenta/music:sequences_lib",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "drums_lib_test",
    srcs = ["drums_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":drums_lib",
        "//magenta/common:testing_lib",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "events_lib",
    srcs = ["events_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/common:sequence_example_lib",
        "//magenta/music:constants",
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
        ":events_lib",
        ":melodies_lib",
        "//magenta/common:testing_lib",
        "//magenta/music:constants",
        "//magenta/music:sequences_lib",
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
        ":lead_sheets_lib",
        ":melodies_lib",
        "//magenta/music:constants",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "melodies_lib",
    srcs = ["melodies_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":events_lib",
        "//magenta/music:constants",
        "//magenta/music:midi_io",
        "//magenta/music:sequences_lib",
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
        ":melodies_lib",
        "//magenta/common:testing_lib",
        "//magenta/music:constants",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)
