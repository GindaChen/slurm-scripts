#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1


module load conda
module load cudatoolkit/12.4
conda activate vllm

srun examples/simple.sh