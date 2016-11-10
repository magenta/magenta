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
# Scripts for preparing data.

package(
    default_visibility = ["//magenta:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "convert_dir_to_note_sequences",
    srcs = ["convert_dir_to_note_sequences.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta/music:midi_io",
        "//magenta/music:musicxml_reader",
        "//magenta/music:note_sequence_io",
        # tensorflow dep
    ],
)

py_test(
    name = "convert_dir_to_note_sequences_test",
    srcs = ["convert_dir_to_note_sequences_test.py"],
    data = ["//magenta/testdata"],
    srcs_version = "PY2AND3",
    deps = [
        ":convert_dir_to_note_sequences",
        # tensorflow dep
    ],
)

py_binary(
    name = "unpack_bundle",
    srcs = ["unpack_bundle.py"],
    deps = [
        "//magenta/music:sequence_generator_bundle",
    ],
)
