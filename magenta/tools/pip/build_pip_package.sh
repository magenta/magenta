#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

function main() {
  if [[ $# -lt 1 ]] ; then
    echo "No destination dir provided."
    echo "Example usage: $0 /tmp/magenta_pkg"
    exit 1
  fi

  DEST="$1"
  TMPDIR="$(mktemp -d)"

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  if [[ ! -d "bazel-bin/magenta" ]]; then
    echo "Could not find bazel-bin. Did you run from the root of the build "\
      "tree?"
    exit 1
  fi

  RUNFILES="bazel-bin/magenta/tools/pip/build_pip_package.runfiles"
  cp -RL "${RUNFILES}/__main__/magenta" "${TMPDIR}/"

  cp magenta/tools/pip/setup.py "${TMPDIR}"

  pushd "${TMPDIR}"
  rm -f MANIFEST
  echo $(date) : "=== Building wheel"
  python setup.py bdist_wheel # >/dev/null
  mkdir -p "${DEST}"
  cp dist/* "${DEST}"
  popd
  rm -rf "${TMPDIR}"
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
