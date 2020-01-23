# Copyright 2020 The Magenta Authors.
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

#!/bin/bash
# Install the .protoc compiler
curl -o /tmp/protoc3.zip -L https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-${1}-x86_64.zip

# Unzip
unzip -d /tmp/protoc3 /tmp/protoc3.zip

# Move protoc to /usr/local/bin/
sudo mv /tmp/protoc3/bin/* /usr/local/bin/

# Move protoc3/include to /usr/local/include/
sudo mv /tmp/protoc3/include/* /usr/local/include/
