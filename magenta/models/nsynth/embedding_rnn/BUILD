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
        "//magenta/models/nsynth/embedding_rnn:__subpackages__",
        "//magenta/tools/pip:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "run",
    srcs = [
        "run.py",
    ],
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_binary(
    name = "embedding_rnn",
    srcs = [
        "embedding_rnn.py",
    ],
    deps = [
        # numpy dep
        # sklearn dep
        # tensorflow dep
    ],
)

py_binary(
    name = "synthesize",
    srcs = [
        "synthesize.py",
    ],
    deps = [
        ":embedding_rnn",
        "//third_party/magenta/models/nsynth/wavenet:fastgen",
        # numpy dep
        # sklearn dep
        # tensorflow dep
    ],
)

py_test(
    name = "embedding_rnn_test",
    srcs = ["embedding_rnn_test.py"],
    deps = [
        ":embedding_rnn",
        # numpy dep
        # tensorflow dep
    ],
)
