#!/bin/bash

if [ -z "$1" ]
then
  echo 'Must provide a name for the virtualenv'
  exit 1
else
  VIRTUALENV_NAME=$1
fi

virtualenv --system-site-packages -p python3 ~/$VIRTUALENV_NAME
source ~/$VIRTUALENV_NAME/bin/activate

pip install .
pip install -r requirements.txt
