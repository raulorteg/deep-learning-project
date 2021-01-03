#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Procgen
#BSUB -n 1
#BSUB -W 03:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/log_%J.out
#BSUB -e logs/log_%J.err

mkdir -p logs
pip3 install procgen --user
echo "Running script..."
python3 Models/IMPALAx4.py



