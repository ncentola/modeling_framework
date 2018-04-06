#!/bin/bash

VIRTUALENV_NAME=$1

virtualenv --system-site-packages -p python3 ~/$VIRTUALENV_NAME
source ~/$VIRTUALENV_NAME/bin/activate

pip install .
pip install -r dev_requirements.txt
