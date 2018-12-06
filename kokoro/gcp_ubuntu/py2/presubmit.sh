#!/bin/bash

# Fail on any error.
set -e
# Display commands to stderr.
set -x

cd github/magenta

sudo apt-get install build-essential libasound2-dev libjack-dev
python2 setup.py test
