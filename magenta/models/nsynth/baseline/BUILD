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

# Description: TF code for training baseline specgram autoencoder models

package(
    default_visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "ae_configs",
    srcs = glob(["models/ae_configs/*.py"]),
    deps = [
        "//magenta/common:tf_lib",
        "//magenta/models/nsynth:utils",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "ae",
    srcs = ["models/ae.py"],
    deps = [
        ":ae_configs",
        "//magenta/common:tf_lib",
        "//magenta/models/nsynth:utils",
        # tensorflow dep
    ],
)

py_library(
    name = "library",
    deps = [
        ":ae",
        "//magenta/models/nsynth:reader",
        "//magenta/models/nsynth:utils",
        # librosa dep
        # tensorflow dep
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":library",
    ],
)

py_binary(
    name = "save_embeddings",
    srcs = ["save_embeddings.py"],
    deps = [
        ":library",
    ],
)

