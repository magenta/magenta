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

# Tensorflow version of Pix2pix.
py_binary(
    name = "main",
    srcs = ["main.py"],
    visibility =  ["//visibility:public"],
    deps = [
        "model",
        "ops",
        "utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)
