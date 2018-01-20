#/bin/bash
set -eo pipefail
IFS=$'\n\t'

env_dir=~/venv/LabeledVarAutoencoder

if [ -n "$(type -t deactivate)" ] && [ "$(type -t deactivate)" = function ]; then
    deactivate
fi
echo "Deleting and re-initalizing virtualenv"
rm -rf "$env_dir"
mkdir -p "$env_dir"

echo "Initializng virtualenv"
python3 -m venv "$env_dir"
. "$env_dir/bin/activate"
echo "Upgrading pip and installing deps"
pip3 install --upgrade pip

# Use
if [ -e /usr/lib/x86_64-linux-gnu/libcudnn.so ]; then
    echo "Using tensorflow gpu"
    tensorflow=tensorflow-gpu
else
    echo "Using non-gpu tensorflow"
    tensorflow=tensorflow
fi
chocolate=git+https://github.com/AIworx-Labs/chocolate@master
pip3 install $tensorflow matplotlib scipy pillow "$chocolate"
echo -e "\nNow run\n\nsource $env_dir/bin/activate"
