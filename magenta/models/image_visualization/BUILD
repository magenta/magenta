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
    name = "visualize_utils",
    srcs = ["visualize_utils.py"],
    deps = [
        # tensorflow dep
        # scipy misc dep
        # skimage restoration dep
        # six iteritems dep
    ],
)

py_binary(
    name = "class_visualization",
    srcs = ["class_visualization.py"],
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":visualize_utils"
        # scipy misc dep
        # tensorflow dep
        # h5py dep
    ],
)
