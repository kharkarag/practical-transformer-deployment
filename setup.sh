#!/bin/bash

# This script assumes host machine has conda and gsutil

git clone https://github.com/kharkarag/practical-transformer-deployment.git
cd practical-transformer-deployment

conda env create -f environment.yml -y
source ~/.bashrc
conda activate project

bash convert.sh

# To run the server after this command, run `export FLASK_APP=inference && flask run --host=0.0.0.0 --port=5001`, preferably in a screen
