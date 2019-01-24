#!/bin/bash

##
# Steps to run CI tests.
##

set -e
set -x

source /tmp/magenta-env/bin/activate

pytest
pylint magenta
