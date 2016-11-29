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
# Libraries for encoding and decoding musical event sequences.

package(
    default_visibility = ["//magenta:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "chords_encoder_decoder",
    srcs = ["chords_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":encoder_decoder",
        "//magenta/music:chord_symbols_lib",
        "//magenta/music:constants",
    ],
)

py_test(
    name = "chords_encoder_decoder_test",
    srcs = ["chords_encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":chords_encoder_decoder",
        "//magenta/music:constants",
        # tensorflow dep
    ],
)

py_library(
    name = "drums_encoder_decoder",
    srcs = ["drums_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":encoder_decoder",
    ],
)

py_test(
    name = "drums_encoder_decoder_test",
    srcs = ["drums_encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":drums_encoder_decoder",
        # tensorflow dep
    ],
)

py_library(
    name = "encoder_decoder",
    srcs = ["encoder_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/common:sequence_example_lib",
        "//magenta/music:constants",
        "//magenta/pipelines",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "encoder_decoder_test",
    srcs = ["encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":encoder_decoder",
        "//magenta/common:sequence_example_lib",
        "//magenta/music:testing_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "melody_encoder_decoder",
    srcs = ["melody_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":encoder_decoder",
        "//magenta/music:constants",
        "//magenta/music/events:events_lib",
        "//magenta/music/events:melodies_lib",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
        # numpy dep
    ],
)

py_test(
    name = "melody_encoder_decoder_test",
    srcs = ["melody_encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":melody_encoder_decoder",
        "//magenta/common:sequence_example_lib",
        "//magenta/music:constants",
        "//magenta/music:sequences_lib",
        "//magenta/music/events:melodies_lib",
        # tensorflow dep
    ],
)
