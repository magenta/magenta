#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# TODO(iansimon): Unrestrict tornado version and figure out how to upgrade to
# python 2.7.9 in kokoro.
eval "${PIP_COMMAND} install --upgrade \
  'tornado<5.0' \
  IPython \
  bokeh \
  intervaltree \
  librosa \
  matplotlib \
  mir_eval \
  scipy \
  tensorflow \
  "

bazel test \
  --keep_going \
  --test_output=errors \
  ${BAZEL_TEST_ARGS} \
  -- \
  //magenta/...
