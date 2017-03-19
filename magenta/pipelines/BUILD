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
    name = "pipelines",
    deps = [
        ":dag_pipeline",
        ":drum_pipelines",
        ":lead_sheet_pipelines",
        ":melody_pipelines",
        ":pipeline",
        ":pipelines_common",
        ":statistics",
    ],
)

py_library(
    name = "chord_pipelines",
    srcs = ["chord_pipelines.py"],
    deps = [
        ":pipeline",
        "//magenta/music:chord_symbols_lib",
        "//magenta/music:chords_lib",
        "//magenta/music:events_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "chord_pipelines_test",
    srcs = ["chord_pipelines_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chord_pipelines",
        "//magenta/common:testing_lib",
        "//magenta/music:chords_lib",
        "//magenta/music:constants",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "dag_pipeline",
    srcs = ["dag_pipeline.py"],
    deps = [
        ":pipeline",
    ],
)

py_test(
    name = "dag_pipeline_test",
    srcs = ["dag_pipeline_test.py"],
    deps = [
        ":dag_pipeline",
        ":pipeline",
        ":statistics",
        # tensorflow dep
    ],
)

py_library(
    name = "drum_pipelines",
    srcs = ["drum_pipelines.py"],
    deps = [
        ":pipeline",
        "//magenta/music:drums_lib",
        "//magenta/music:events_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "drum_pipelines_test",
    srcs = ["drum_pipelines_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":drum_pipelines",
        "//magenta/common:testing_lib",
        "//magenta/music:drums_lib",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "lead_sheet_pipelines",
    srcs = ["lead_sheet_pipelines.py"],
    deps = [
        ":pipeline",
        "//magenta/music:chord_symbols_lib",
        "//magenta/music:events_lib",
        "//magenta/music:lead_sheets_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "lead_sheet_pipelines_test",
    srcs = ["lead_sheet_pipelines_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lead_sheet_pipelines",
        "//magenta/common:testing_lib",
        "//magenta/music:chords_lib",
        "//magenta/music:constants",
        "//magenta/music:lead_sheets_lib",
        "//magenta/music:melodies_lib",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "melody_pipelines",
    srcs = ["melody_pipelines.py"],
    deps = [
        ":pipeline",
        "//magenta/music:events_lib",
        "//magenta/music:melodies_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "melody_pipelines_test",
    srcs = ["melody_pipelines_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melody_pipelines",
        "//magenta/common:testing_lib",
        "//magenta/music:constants",
        "//magenta/music:melodies_lib",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":statistics",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
    data = [
        "//magenta/testdata:tfrecord_iterator_test.tfrecord",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":pipeline",
        "//magenta/common:testing_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "pipelines_common",
    srcs = ["pipelines_common.py"],
    deps = [
        ":pipeline",
        "//magenta/music:sequences_lib",
        "//magenta/protobuf:music_py_pb2",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "pipelines_common_test",
    srcs = ["pipelines_common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pipelines_common",
        "//magenta/common:testing_lib",
        "//magenta/music:sequences_lib",
        "//magenta/music:testing_lib",
        "//magenta/protobuf:music_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "statistics",
    srcs = ["statistics.py"],
)

py_test(
    name = "statistics_test",
    srcs = ["statistics_test.py"],
    deps = [
        ":statistics",
        # tensorflow dep
    ],
)
