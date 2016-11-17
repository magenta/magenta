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

py_library(
    name = "polyphony_lib",
    srcs = ["polyphony_lib.py"],
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "polyphony_lib_test",
    srcs = ["polyphony_lib_test.py"],
    deps = [
        ":polyphony_lib",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "polyphony_model",
    srcs = ["polyphony_model.py"],
    deps = [
        ":polyphony_encoder_decoder",
        "//magenta/models/shared:events_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "polyphony_encoder_decoder",
    srcs = ["polyphony_encoder_decoder.py"],
    deps = [
        ":polyphony_lib",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "polyphony_encoder_decoder_test",
    srcs = ["polyphony_encoder_decoder_test.py"],
    deps = [
        ":polyphony_encoder_decoder",
        ":polyphony_lib",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "polyphonic_rnn_create_dataset",
    srcs = ["polyphonic_rnn_create_dataset.py"],
    deps = [
        ":polyphony_lib",
        ":polyphony_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "polyphonic_rnn_create_dataset_test",
    srcs = ["polyphonic_rnn_create_dataset_test.py"],
    deps = [
        ":polyphony_lib",
        ":polyphonic_rnn_create_dataset",
        "//magenta/models/shared:events_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)
