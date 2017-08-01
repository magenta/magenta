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
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "h512_bo16",
    srcs = ["h512_bo16.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
        "//magenta/models/nsynth:reader",
        "//magenta/models/nsynth:utils",
        "//magenta/models/nsynth/wavenet:masked",
    ],
)

py_library(
    name = "fastgen",
    srcs = ["fastgen.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        # numpy dep
        # scipy dep
        # tensorflow dep
        "//magenta/models/nsynth:utils",
        "//magenta/models/nsynth/wavenet:h512_bo16",
    ],
)

py_library(
    name = "config_library",
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":config_library",
    ],
)

py_binary(
    name = "nsynth_generate",
    srcs = [
        "nsynth_generate.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":configs",
        ":fastgen",
        "//magenta/models/nsynth:utils",
        # tensorflow dep
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":configs",
        "//magenta/models/nsynth:utils",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "nsynth_save_embeddings",
    srcs = [
        "nsynth_save_embeddings.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":configs",
        ":fastgen",
        "//magenta/models/nsynth:reader",
        "//magenta/models/nsynth:utils",
        # numpy dep
        # tensorflow dep
    ],
)
