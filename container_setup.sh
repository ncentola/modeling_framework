#!/bin/bash

VIRTUALENV_NAME=$1

virtualenv --system-site-packages -p python3 ~/virtualenvs/$VIRTUALENV_NAME
source ~/virtualenvs/$VIRTUALENV_NAME/bin/activate

pip install .
pip install -r dev_requirements.txt
