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

# Description: Neural synthesizer model.

package(
    default_visibility = [
        "//magenta/models/nsynth:__subpackages__",
        "//magenta/tools/pip:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/models/nsynth:__subpackages__",
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        # librosa dep
        # numpy dep
        # tensorflow dep
        # scipy dep
    ],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # numpy dep
        # tensorflow dep
    ],
)
