# Copyright 2018 Google Inc. All Rights Reserved.
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

package(
    default_visibility = ["//magenta:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "problems",
    srcs = ["problems.py"],
)

py_binary(
    name = "t2t_datagen",
    srcs = ["t2t_datagen.py"],
    deps = [
        ":problems",
        # tensor2tensor/bin:t2t_datagen_bare dep
    ],
)

py_binary(
    name = "t2t_decoder",
    srcs = ["t2t_decoder.py"],
    deps = [
        ":problems",
        # tensor2tensor/bin:t2t_decoder_lib dep
    ],
)

py_binary(
    name = "t2t_trainer",
    srcs = ["t2t_trainer.py"],
    deps = [
        ":problems",
        # tensor2tensor/bin:t2t_trainer_bare dep
    ],
)