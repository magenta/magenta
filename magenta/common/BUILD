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
    name = "common",
    srcs = ["__init__.py"],
    deps = [
        ":concurrency",
        ":sequence_example_lib",
        ":testing_lib",
        ":tf_lib",
    ],
)

py_library(
    name = "concurrency",
    srcs = ["concurrency.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "concurrency_test",
    srcs = ["concurrency_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":concurrency",
        # tensorflow dep
    ],
)

py_library(
    name = "sequence_example_lib",
    srcs = ["sequence_example_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "testing_lib",
    srcs = ["testing_lib.py"],
    deps = [
        "@protobuf//:protobuf_python",
        # numpy dep
    ],
)

py_library(
    name = "tf_lib",
    srcs = ["tf_lib.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "tf_lib_test",
    srcs = ["tf_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tf_lib",
        # tensorflow dep
    ],
)

