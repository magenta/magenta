# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@tf//google/protobuf:protobuf.bzl", "cc_proto_library")
load("@tf//google/protobuf:protobuf.bzl", "py_proto_library")

def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.
    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.
    """
    return select({
        "@tf//third_party/gpus/cuda:using_nvcc": if_true,
        "@tf//third_party/gpus/cuda:using_gcudacc": if_true,
        "//conditions:default": if_false
    })

def tf_copts():
  return (["-fno-exceptions", "-DEIGEN_AVOID_STL_ARRAY",] +
          if_cuda(["-DGOOGLE_CUDA=1"]) +
          select({"@tf//tensorflow:darwin": [],
                  "//conditions:default": ["-pthread"]}))

def tf_proto_library(name, srcs=[], has_services=False,
                     deps=[], visibility=None, testonly=0,
                     cc_api_version=2, go_api_version=2,
                     java_api_version=2,
                     py_api_version=2):
  native.filegroup(name=name + "_proto_srcs",
                   srcs=srcs,
                   testonly=testonly,)

  cc_proto_library(name=name,
                   srcs=srcs,
                   deps=deps,
                   cc_libs = ["@tf//google/protobuf:protobuf"],
                   protoc="@tf//google/protobuf:protoc",
                   default_runtime="@tf//google/protobuf:protobuf",
                   testonly=testonly,
                   visibility=visibility,)

def tf_proto_library_py(name, srcs=[], deps=[], visibility=None, testonly=0):
  py_proto_library(name=name,
                   srcs=srcs,
                   srcs_version = "PY2AND3",
                   deps=deps,
                   default_runtime="@tf//google/protobuf:protobuf_python",
                   protoc="@tf//google/protobuf:protoc",
                   visibility=visibility,
                   testonly=testonly,)
