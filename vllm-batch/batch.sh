#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1

echo "Starting script"
pwd

echo "Loading modules"

module load conda
module load cudatoolkit/12.4
conda activate vllm


echo "Start Execution"

srun examples/simple.sh
srun examples/simple_template.sh
