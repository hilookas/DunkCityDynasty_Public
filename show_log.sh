#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dunk_city_dynasty
pkill tensorboard
nohup tensorboard --bind_all --logdir $1 &