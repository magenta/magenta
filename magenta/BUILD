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

package(
    default_visibility = ["//visibility:private"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

load(
    "magenta",
    "tf_proto_library",
    "tf_proto_library_py",
)

tf_proto_library(
    name = "music_proto",
    srcs = ["protobuf/music.proto"],
)

tf_proto_library_py(
    name = "music_py_pb3",
    srcs = ["protobuf/music.proto"],
)