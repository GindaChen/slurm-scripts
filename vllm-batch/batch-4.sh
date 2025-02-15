#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00

echo "Starting script"
pwd

echo "Loading modules"

module load conda
module load cudatoolkit/12.4
conda activate vllm


echo "Start Execution"

bash examples/simple.sh
bash examples/simple_template.sh
