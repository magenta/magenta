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
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
    srcs_version = "PY2AND3",
    data = [
        "testdata/tfrecord_iterator_test.tfrecord",
    ],
    deps = [
        ":pipeline",
        # tensorflow dep
    ],
)

py_library(
    name = "pipeline_unit",
    srcs = ["pipeline_unit.py"],
)

py_library(
    name = "pipeline_units_common",
    srcs = ["pipeline_units_common.py"],
    deps = [
        ":pipeline_unit",
        "//magenta/lib:melodies_lib",
        "//magenta/lib:sequences_lib",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "pipeline_units_common_test",
    srcs = ["pipeline_units_common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pipeline_unit",
        ":pipeline_units_common",
        "//magenta/lib:melodies_lib",
        "//magenta/lib:sequences_lib",
        "//magenta/lib:testing_lib",
        # tensorflow dep
    ],
)