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
    srcs_version = "PY2AND3",
    data = [
        "//magenta/testdata:tfrecord_iterator_test.tfrecord",
    ],
    deps = [
        ":pipeline",
        "//magenta/lib:testing_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "pipelines_common",
    srcs = ["pipelines_common.py"],
    deps = [
        ":pipeline",
        "//magenta/lib:melodies_lib",
        "//magenta/lib:sequences_lib",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "pipelines_common_test",
    srcs = ["pipelines_common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pipelines_common",
        "//magenta/lib:melodies_lib",
        "//magenta/lib:sequences_lib",
        "//magenta/lib:testing_lib",
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