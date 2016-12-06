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

# Description: An image style transfer model.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "imagenet_data",
    srcs = ["imagenet_data.py"],
    deps = [
        # tensorflow dep
    ],
)

py_binary(
    name = "image_stylization_create_dataset",
    srcs = ["image_stylization_create_dataset.py"],
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":image_utils",
        ":learning",
        # scipy dep
        # scipy pilutil dep
        # tensorflow dep
    ],
)

py_binary(
    name = "image_stylization_evaluate",
    srcs = ["image_stylization_evaluate.py"],
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":image_utils",
        ":learning",
        ":model",
    ],
)

py_binary(
    name = "image_stylization_finetune",
    srcs = ["image_stylization_finetune.py"],
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":image_utils",
        ":learning",
        ":model",
        ":vgg",
    ],
)

py_binary(
    name = "image_stylization_train",
    srcs = ["image_stylization_train.py"],
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":image_utils",
        ":learning",
        ":model",
        ":vgg",
    ],
)

py_binary(
    name = "image_stylization_transform",
    srcs = ["image_stylization_transform.py"],
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":image_utils",
        ":model",
    ],
)

py_library(
    name = "image_utils",
    srcs = ["image_utils.py"],
    data = [
        "evaluation_images",
    ],
    deps = [
        ":imagenet_data",
        # numpy dep
        # storage dep
        # scipy dep
        # scipy pilutil dep
        # tensorflow dep
    ],
)

py_library(
    name = "learning",
    srcs = ["learning.py"],
    deps = [
        ":vgg",
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
        ":ops",
        # tensorflow dep
    ],
)

py_library(
    name = "ops",
    srcs = [
        "ops.py",
    ],
    deps = [
        # tensorflow dep
    ],
)

py_library(
    name = "vgg",
    srcs = [
        "vgg.py",
    ],
    deps = [
        # tensorflow dep
    ],
)
