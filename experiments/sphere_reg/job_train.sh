#!/bin/bash
#BSUB -J sphere_reg_train
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -W 1:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o /home/s252125/my_project/logs/train_%J.out
#BSUB -e /home/s252125/my_project/logs/train_%J.err

source $HOME/miniconda3/bin/activate
conda activate brainnet

cd /home/s252125/my_project
python -m experiments.sphere_reg.train_small