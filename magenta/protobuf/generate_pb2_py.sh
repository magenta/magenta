#!/bin/bash

# This script use the protoc compiler to generate the python code of the
# .proto files.

if [[ $(protoc --version) != 'libprotoc 3.6.1' ]]; then
  echo 'Please use version 3.6.1 of protoc for compatibility with Python 2 and 3.'
  exit
fi

# Make it possible to run script from project root dir:
cd `dirname $0`

function gen_proto {
  echo "gen_proto $1..."
  protoc $1.proto --python_out=.
  # We don't want pylint to run on this file, so we prepend directives.
  printf "%s\n%s" "# pylint: skip-file" "$(cat $1_pb2.py)" > \
    $1_pb2.py
  echo "done"
}

gen_proto generator
gen_proto music
