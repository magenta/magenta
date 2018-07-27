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

# A script to generate videos based on next frame prediction.
sh_binary(
    name = "create_video",
    srcs = ["create_video.sh"],
    data = [
        "join_pairs",
        "recursion_640",
        "//magenta/video/tools:convert2jpg",
        "//magenta/video/tools:extract_frames",
        "//magenta/video/tools:random_pick",
        "@pix2pix_tensorflow//:main",
    ],
    visibility = ["//magenta:__subpackages__"],
)

# A script to use pix2pix recursivly, with 640x360 video
sh_binary(
    name = "recursion_640",
    srcs = ["recursion_640.sh"],
    visibility = ["//magenta:__subpackages__"],
)

# A script to assemble 2 images in 1, to create pairs to be processed by pix2pix
py_library(
    name = "join_pairs",
    srcs = ["join_pairs.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta:__subpackages__"],
)
