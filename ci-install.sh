#!/bin/bash

##
# Steps needed to set up CI environment.
##

set -e
set -x

sudo apt-get update
sudo apt-get -y install build-essential libasound2-dev libjack-dev libav-tools

# Ensure python 3.5 is used, set up an isolated virtualenv.
PY3_PATH="$(which python3.5)"
${PY3_PATH} -m virtualenv /tmp/magenta-env --python="${PY3_PATH}"
source /tmp/magenta-env/bin/activate
echo $(which python)
python --version

pip install pylint pytest
python setup.py bdist_wheel --universal
pip install --upgrade --ignore-installed dist/magenta*.whl
