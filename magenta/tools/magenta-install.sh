# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
#
#
# An install script for magenta (https://github.com/tensorflow/magenta).
# Run with: bash install_magenta.sh

# Exit on error
set -e

finish() {
  if (( $? != 0)); then
    echo ""
    echo "==========================================="
    echo "Installation did not finish successfully."
    echo "Please follow the manual installation instructions at:"
    echo "https://github.com/tensorflow/magenta"
    echo "==========================================="
    echo ""
  fi
}
trap finish EXIT

# For printing error messages
err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
  exit 1
}

# Check which operating system
if [[ "$(uname)" == "Darwin" ]]; then
    echo 'Mac OS Detected'
    readonly OS='MAC'
    readonly MINICONDA_SCRIPT='Miniconda3-latest-MacOSX-x86_64.sh'
elif [[ "$(uname)" == "Linux" ]]; then
    echo 'Linux OS Detected'
    readonly OS='LINUX'
    readonly MINICONDA_SCRIPT='Miniconda3-latest-Linux-x86_64.sh'
else
    err 'Detected neither OSX or Linux Operating System'
fi

# Check if anaconda already installed
if [[ ! $(which conda) ]]; then
    echo ""
    echo "==========================================="
    echo "anaconda not detected, installing miniconda"
    echo "==========================================="
    echo ""
    readonly CONDA_INSTALL="/tmp/${MINICONDA_SCRIPT}"
    readonly CONDA_PREFIX="${HOME}/miniconda3"
    curl "https://repo.continuum.io/miniconda/${MINICONDA_SCRIPT}" > "${CONDA_INSTALL}"
    bash "${CONDA_INSTALL}" -p "${CONDA_PREFIX}"
    # Modify the path manually rather than sourcing .bashrc because some .bashrc
    # files refuse to execute if run in a non-interactive environment.
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    if [[ ! $(which conda) ]]; then
      err 'Could not find conda command. conda binary was not properly added to PATH'
    fi
else
    echo ""
    echo "========================================="
    echo "anaconda detected, skipping conda install"
    echo "========================================="
    echo ""
fi

# Set up the magenta environment
echo ""
echo "=============================="
echo "setting up magenta environment"
echo "=============================="
echo ""

conda create -n magenta python=3.7

# Need to deactivate set -e because the conda activate script was not written
# with set -e in mind, and because we source it here, the -e stays active.
# In order to determine if any errors occurred while executing it, we verify
# that the environment changed afterward.
set +e
conda activate magenta
set -e
if [[ $(conda info --envs | grep "*" | awk '{print $1}') != "magenta" ]]; then
  err 'Did not successfully activate the magenta conda environment'
fi

# Install other dependencies
pip install jupyter magenta

# Install rtmidi for realtime midi IO
if [[ $(which apt-get) ]]; then
    echo ""
    echo "============================================"
    echo "Installing rtmidi Linux library dependencies"
    echo "sudo privileges required"
    echo "============================================"
    echo ""
    sudo apt-get install build-essential libasound2-dev libjack-dev
fi
pip install --pre python-rtmidi

echo ""
echo "=============================="
echo "Magenta Install Success!"
echo ""
echo "NOTE:"
echo "For changes to become active, you will need to open a new terminal."
echo ""
echo "For complete uninstall, remove the installed anaconda directory:"
echo "rm -r ~/miniconda2"
echo ""
echo "To just uninstall the environment run:"
echo "conda remove -n magenta --all"
echo ""
echo "To run magenta, activate your environment:"
echo "source activate magenta"
echo ""
echo "You can deactivate when you're done:"
echo "source deactivate"
echo "=============================="
echo ""
