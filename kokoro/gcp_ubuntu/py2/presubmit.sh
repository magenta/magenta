#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

# Ensure that python 2 is used.
PY2_PATH="$(which python2.7.9)"
export PIP_COMMAND="sudo ${PY2_PATH} -m pip"
export BAZEL_TEST_ARGS="--force_python=py2 --python_path=${PY2_PATH}"

cd github/magenta
kokoro/test.sh
