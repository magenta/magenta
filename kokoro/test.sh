#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

bazel test --test_lang_filters=py -k \
      --test_output=errors -- \
          //magenta/...
