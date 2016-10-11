#!/bin/bash
#
# An install script for magenta (https://github.com/tensorflow/magenta).
# Run with: bash install_magenta.sh

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

# Exit on error
set -e

# For printing error messages
err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
  exit 1
}

# Check which operating system
if [[ "$(uname)" == "Darwin" ]]; then
    echo 'Mac OS Detected'
    readonly OS='MAC'
    readonly MINICONDA_SCRIPT='Miniconda2-latest-MacOSX-x86_64.sh'
    # Mac OS X, CPU only, Python 2.7:
    readonly TF_BINARY_URL='https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc0-py2-none-any.whl'
elif [[ "$(uname)" == "Linux" ]]; then
    echo 'Linux OS Detected'
    readonly OS='LINUX'
    readonly MINICONDA_SCRIPT='Miniconda2-latest-Linux-x86_64.sh'
    # Ubuntu/Linux 64-bit, CPU only, Python 2.7
    readonly TF_BINARY_URL='https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl'
else
    err 'Detected neither OSX or Linux Operating System'
fi

echo ${MINICONDA_SCRIPT}

# Check if anaconda already installed
if [[ ! $(which conda) ]]; then
    echo ""
    echo "==========================================="
    echo "anaconda not detected, installing miniconda"
    echo "==========================================="
    echo ""
    readonly CONDA_INSTALL="/tmp/${MINICONDA_SCRIPT}"
    curl "https://repo.continuum.io/miniconda/${MINICONDA_SCRIPT}" > "${CONDA_INSTALL}"
    bash $CONDA_INSTALL
    # Miniconda installer appends to path differently for different OS
    if [[ $OS == "LINUX" ]]; then
        source "${HOME}/.bashrc"
    elif [[ $OS == "MAC" ]]; then
        source "${HOME}/.bash_profile"
    fi
else
    echo ""
    echo "==================================="
    echo "anaconda detected, skipping install"
    echo "==================================="
    echo ""
fi

# Set up the magenta environment
echo ""
echo "=============================="
echo "setting up magenta environment"
echo "=============================="
echo ""

conda create -n magenta python=2.7
source activate magenta

# Install tensorflow
pip install --ignore-installed --upgrade $TF_BINARY_URL

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
echo "For complete uninstall, remove the installed anaconda directory:"
echo "rm -r ~/miniconda2"
echo ""
echo "To just uninstall the environment run:"
echo "conda remove -n magenta --all"
echo ""
echo "To run magenta activate your environment:"
echo "source activate magenta"
echo ""
echo "You can deactivate when you're done:"
echo "source deactivate"
echo "=============================="
echo ""

