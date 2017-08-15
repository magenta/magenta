#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

export BAZEL_TEST_ARGS='--force_python=py3  --test_tag_filters=-py2only'

cd github/magenta
kokoro/test.sh
