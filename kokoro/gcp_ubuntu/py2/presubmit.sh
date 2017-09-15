#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

# Ensure that python 2 is used.
export PIP_COMMAND='sudo python2 -m pip'
export BAZEL_TEST_ARGS='--force_python=py2'

cd github/magenta
kokoro/test.sh
