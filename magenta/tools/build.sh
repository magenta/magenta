#!/bin/bash
#
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
#
# Builds the pip package, and then installs it.
# Usage: bash build.sh

# Exit on error
set -e

# Get script directory
readonly DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#In case user is outside of magenta dir
cd "${DIR}/../.."

# build the binary to make pip package
bazel build //magenta/tools/pip:build_pip_package

# Make sure there is no tmp directory
if [[ -e /tmp/magenta_pkg ]]; then
    rm -r /tmp/magenta_pkg
fi

# Make a pip package. Script should be in //magenta/tools
bazel-bin/magenta/tools/pip/build_pip_package /tmp/magenta_pkg
pip install -U /tmp/magenta_pkg/*
