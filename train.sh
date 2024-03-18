#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dunk_city_dynasty
mkdir -p logs
screen -L -Logfile 'logs/train.log' python ./baselines/train_rl_2.py