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

licenses(["notice"])  # Apache 2.0

# The Magenta public API.
py_library(
    name = "magenta",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta:__subpackages__"],
    deps = [
        ":version",
        "//magenta/common",
        "//magenta/music",
        "//magenta/pipelines",
        "//magenta/protobuf",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY2AND3",
)
