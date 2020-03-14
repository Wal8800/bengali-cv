#!/usr/bin/env bash

sudo apt-get install libssl-dev
sudo apt-get install libsqlite3-dev
sudo apt-get install libffi-dev
sudo apt-get install libbz2-dev

pip install --user pipenv
export PATH=$PATH:~/.local/bin

curl https://pyenv.run | bash


# export PATH="~/.pyenv/bin:$PATH"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"