# Copyright 2018 Google Inc. All Rights Reserved.
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

# Description: A simple interface to control the piano.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "configs",
    srcs = ["configs.py"],
)

py_library(
    name = "loader",
    srcs = ["loader.py"],
    deps = [
        "//magenta/protobuf",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    deps = [
        ":util",
        # sonnet dep
        # tensorflow dep
    ],
)

py_library(
    name = "gold",
    srcs = [
        "gold.py",
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "eval",
    srcs = [
        "eval.py",
    ],
    visibility = [
        # internal model:piano_genie
    ],
    deps = [
        ":configs",
        ":gold",
        ":loader",
        ":model",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    visibility = [
        # internal model:piano_genie
    ],
    deps = [
        ":configs",
        ":loader",
        ":model",
        ":util",
        # tensorflow dep
    ],
)
