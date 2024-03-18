#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dunk_city_dynasty
mkdir -p logs_bc
screen -L -Logfile 'logs_bc/train.log' python ./baselines/train_bc_2.py