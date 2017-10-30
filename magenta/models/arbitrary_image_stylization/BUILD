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

# Description: An image style transfer model.

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "arbitrary_image_stylization_evaluate",
    srcs = ["arbitrary_image_stylization_evaluate.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":arbitrary_image_stylization_build_model",
        "//magenta/models/image_stylization:image_utils",
        "//magenta/models/image_stylization:learning",
    ],
)

py_binary(
    name = "arbitrary_image_stylization_train",
    srcs = ["arbitrary_image_stylization_train.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":arbitrary_image_stylization_build_model",
        "//magenta/models/image_stylization:image_utils",
        "//magenta/models/image_stylization:learning",
        "//magenta/models/image_stylization:vgg",
    ],
)

py_binary(
    name = "arbitrary_image_stylization_with_weights",
    srcs = ["arbitrary_image_stylization_with_weights.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":arbitrary_image_stylization_build_model",
        "//magenta/models/image_stylization:image_utils",
    ],
)

py_library(
    name = "arbitrary_image_stylization_losses",
    srcs = ["arbitrary_image_stylization_losses.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/image_stylization:learning",
        "//magenta/models/image_stylization:vgg",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "nza_model",
    srcs = [
        "nza_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/image_stylization:model",
        "//magenta/models/image_stylization:ops",
        # tensorflow dep
    ],
)

py_library(
    name = "arbitrary_image_stylization_build_model",
    srcs = [
        "arbitrary_image_stylization_build_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/models/image_stylization:ops",
        ":nza_model",
        ":arbitrary_image_stylization_losses",
        # tensorflow dep
    ],
)

