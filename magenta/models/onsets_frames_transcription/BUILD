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

licenses(["notice"])  # Apache 2.0

py_library(
    name = "onsets_frames_transcription",
    srcs = ["__init__.py"],
    visibility = [
        # internal notebook binary
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":constants",
        ":data",
        ":infer_util",
        ":model",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        "//magenta",
        # tensorflow dep
        # slim dep
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY2AND3",
    deps = [
        # librosa dep
        # tensorflow dep
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # librosa dep
        # numpy dep
        # tensorflow dep
        # slim dep
    ],
)

py_test(
    name = "data_test",
    srcs = ["data_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":data",
        "//magenta",
        # tensorflow dep
        # numpy dep
        # slim dep
    ],
)

py_library(
    name = "train_util",
    srcs = ["train_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        ":data",
        ":infer_util",
        ":model",
        "@pretty_midi",
        # mir_eval dep
        # tensorflow dep
        # slim dep
    ],
)

py_library(
    name = "infer_util",
    srcs = ["infer_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":data",
        "//magenta",
        "@pretty_midi",
        # mir_eval dep
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "infer_util_test",
    srcs = ["infer_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":infer_util",
        "//magenta",
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "onsets_frames_transcription_infer",
    srcs = ["onsets_frames_transcription_infer.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        "//magenta",
        ":constants",
        ":data",
        ":infer_util",
        ":model",
        # librosa dep
        # numpy dep
        # scipy dep
        # slim dep
        # tensorflow dep
    ],
)

py_binary(
    name = "onsets_frames_transcription_transcribe",
    srcs = ["onsets_frames_transcription_transcribe.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        "//magenta",
        ":constants",
        ":data",
        ":infer_util",
        ":model",
        # librosa dep
        # tensorflow dep
    ],
)

py_binary(
    name = "onsets_frames_transcription_train",
    srcs = ["onsets_frames_transcription_train.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":model",
        ":train_util",
        # tensorflow dep
    ],
)

py_binary(
    name = "onsets_frames_transcription_create_dataset",
    srcs = ["onsets_frames_transcription_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        "//magenta",
        # tensorflow dep
        # librosa dep
        # numpy dep
    ],
)

py_test(
    name = "onsets_frames_transcription_create_dataset_test",
    srcs = ["onsets_frames_transcription_create_dataset_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":onsets_frames_transcription_create_dataset",
        "//magenta",
        # tensorflow dep
        # numpy dep
    ],
)
