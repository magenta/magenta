# Copyright 2017 Google Inc. All Rights Reserved.
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

# Description: TF code for training wavenet autoencoder models

package(
    default_visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "masked",
    srcs = ["masked.py"],
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "config_library",
    deps = [
        "//magenta/models/nsynth:reader",
        ":masked",
        # tensorflow dep
    ],
)

py_library(
    name = "configs",
    srcs = [
        "h512_bo16.py",
    ],
    deps = [
        ":config_library",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        ":configs",
        "//magenta/models/nsynth:utils",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "save_embeddings",
    srcs = [
        "save_embeddings.py",
    ],
    deps = [
        ":configs",
        "//magenta/models/nsynth:reader",
        "//magenta/models/nsynth:utils",
        # numpy dep
        # tensorflow dep
    ],
)
