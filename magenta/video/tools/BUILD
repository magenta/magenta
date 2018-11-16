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
# Tools for making videos.

licenses(["notice"])  # Apache 2.0

# A script to concatenate multiple mp4 into one.
sh_binary(
    name = "concat_mp4",
    srcs = ["concat_mp4.sh"],
    visibility = ["//magenta:__subpackages__"],
)

# A script to create an mp4 from jpg
sh_binary(
    name = "create_mp4",
    srcs = ["create_mp4.sh"],
    visibility = ["//magenta:__subpackages__"],
)

# A script to generate videos based on next frame prediction.
sh_binary(
    name = "extract_multiple_video",
    srcs = ["extract_multiple_video.sh"],
    visibility = ["//magenta:__subpackages__"],
)

# a python script to convert jpg files
py_binary(
    name = "convert2jpg",
    srcs = ["convert2jpg.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta:__subpackages__"],
)

# a python script to extract the frames of a video
py_binary(
    name = "extract_frames",
    srcs = ["extract_frames.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta:__subpackages__"],
)

# a python script to randomly pick a file from a given set
py_binary(
    name = "random_pick",
    srcs = ["random_pick.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta:__subpackages__"],
)
