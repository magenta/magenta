#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

# Ensure that python 3 is used.
# Filter out tests that support only python 2.
export PIP='pip3'
export BAZEL_TEST_ARGS='--force_python=py3  --test_tag_filters=-py2only --build_tag_filters=-py2only'

cd github/magenta
kokoro/test.sh
