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

# A scripft to generate videos based on next frame prediction.
sh_binary(
    name = "create_video",
    srcs = ["create_video.sh"],
    visibility = ["//magenta:__subpackages__"],
    data = [ "join_pairs",
             "create_pairs",
             "recursion",
             "recursion_640",
             "//magenta/video/tools:extract_frames",
             "//magenta/video/tools:random_pick",
             "//magenta/video/tools:convert2jpg",
             "//third_party/pix2pix_tensorflow:main"
    ],
)

sh_binary(
    name = "recursion_640",
    srcs = ["recursion_640.sh"],
    visibility = ["//magenta:__subpackages__"],
    deps = [
    ],
)

sh_binary(
    name = "recursion",
    srcs = ["recursion.sh"],
    visibility = ["//magenta:__subpackages__"],
    deps = [
    ],
)

py_library(
    name = "join_pairs",
    srcs = ["join_pairs.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
)

py_binary(
    name = "create_pairs",
    srcs = ["create_pairs.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        "//magenta:version",
    ],
)
