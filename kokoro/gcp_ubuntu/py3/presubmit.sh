#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

# Ensure python 3.5 is used.
PY3_PATH="$(which python3.5)"
export PIP_COMMAND="sudo ${PY3_PATH} -m pip"
# Filter out tests that support only python 2.
export BAZEL_TEST_ARGS="--force_python=py3  --test_tag_filters=-py2only \
  --build_tag_filters=-py2only --python_path=${PY3_PATH}"

cd github/magenta
kokoro/test.sh
