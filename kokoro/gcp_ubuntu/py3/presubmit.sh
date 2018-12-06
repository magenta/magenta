#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

# Ensure python 3.5 is used.
PY3_PATH="$(which python3.5)"

cd github/magenta

$PY3_PATH setup.py test --adopts="--pylint -m pylint"
$PY3_PATH setup.py test
