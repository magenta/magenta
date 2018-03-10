#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

eval "${PIP_COMMAND} install --upgrade tensorflow scipy matplotlib \
  intervaltree bokeh IPython librosa mir_eval 'tornado<5.0'"

bazel test \
  --keep_going \
  --test_output=errors \
  ${BAZEL_TEST_ARGS} \
  -- \
  //magenta/...
