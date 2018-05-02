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

# Description: A variational autoencoder model for musical sequences.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "music_vae",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        # internal model:music_vae
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":base_model",
        ":configs",
        ":data",
        ":data_hierarchical",
        ":lstm_models",
        ":lstm_utils",
        ":trained_model",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "data_test",
    srcs = ["data_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "data_hierarchical",
    srcs = ["data_hierarchical.py"],
    deps = [
        ":data",
        "//magenta",
        # numpy dep
    ],
)

py_test(
    name = "data_hierarchical_test",
    srcs = ["data_hierarchical_test.py"],
    deps = [
        ":data_hierarchical",
        "//magenta",
        # tensorflow dep
    ],
)
py_library(
    name = "configs",
    srcs = ["configs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data",
        ":data_hierarchical",
        ":lstm_models",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "base_model",
    srcs = ["base_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "lstm_utils",
    srcs = ["lstm_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
    ],
)

py_test(
    name = "lstm_utils_test",
    srcs = ["lstm_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lstm_utils",
    ],
)

py_library(
    name = "lstm_models",
    srcs = ["lstm_models.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_model",
        ":lstm_utils",
    ],
)

py_library(
    name = "trained_model",
    srcs = ["trained_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "music_vae_train",
    srcs = ["music_vae_train.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:music_vae
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":configs",
        ":data",
        ":lstm_models",
    ],
)
