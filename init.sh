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
pip3 install tensorflow matplotlib scipy pillow
echo -e "\nNow run\n\nsource $env_dir/bin/activate"
