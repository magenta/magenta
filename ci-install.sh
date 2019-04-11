# Copyright 2019 The Magenta Authors.
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

##
# Steps needed to set up CI environment.
##

set -e
set -x

sudo apt-get update
sudo apt-get -y install build-essential libasound2-dev libjack-dev libav-tools sox

# Ensure python 3.5 is used, set up an isolated virtualenv.
PY3_PATH="$(which python3.5)"
${PY3_PATH} -m virtualenv /tmp/magenta-env --python="${PY3_PATH}"
source /tmp/magenta-env/bin/activate
echo $(which python)
python --version

python setup.py bdist_wheel --universal
pip install --upgrade --ignore-installed dist/magenta*.whl
