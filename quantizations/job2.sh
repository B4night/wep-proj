#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1

python ./test_performance_resnet18.py