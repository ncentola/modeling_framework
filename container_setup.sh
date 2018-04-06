#!/bin/bash

if [ -z "$1" ]
then
  VIRTUALENV_NAME='python_fraud_model'
else
  VIRTUALENV_NAME=$1
fi

virtualenv --system-site-packages -p python3 ~/$VIRTUALENV_NAME
source ~/$VIRTUALENV_NAME/bin/activate

pip install .
pip install -r requirements.txt
