#!/bin/bash

# Check which operating system
if [ "$(uname)" == "Darwin" ]; then
    echo "Mac OS Detected"
    export OS="MAC"
    export MINICONDA_SCRIPT=Miniconda2-latest-MacOSX-x86_64.sh
    # Mac OS X, CPU only, Python 2.7:
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc0-py2-none-any.whl
elif [ "$(uname)" == "Linux" ]; then
    echo "Linux OS Detected"
    export OS="LINUX"
    export MINICONDA_SCRIPT=Miniconda2-latest-Linux-x86_64.sh
    # Ubuntu/Linux 64-bit, CPU only, Python 2.7
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
fi


# Check if anaconda already installed
if [ ! $(which conda) ]; then
    echo ""
    echo "==========================================="
    echo "anaconda not detected, installing miniconda"
    echo "==========================================="
    echo ""
    curl https://repo.continuum.io/miniconda/$MINICONDA_SCRIPT > $/tmp/$MINICONDA_SCRIPT
    bash $/tmp/$MINICONDA_SCRIPT
    source $HOME/.bash_profile
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
if [ $OS == "LINUX" ]; then
    echo ""
    echo "============================================"
    echo "Installing rtmidi Linux library dependencies"
    echo "Sudo privileges required"
    echo "============================================"
    echo ""
    sudo apt-get install sudo apt-get install build-essential
    sudo apt-get install libasound2-dev 
    sudo apt-get install libjack-dev
fi
pip install python-rtmidi

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

