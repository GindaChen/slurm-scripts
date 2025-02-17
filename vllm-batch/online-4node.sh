#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=1
#SBATCH --nodes=4
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err


echo "Starting script"
pwd

echo "Loading modules"

module load conda
module load cudatoolkit/12.4
conda activate vllm

# vLLM Configuration
export PIPELINE_PARALLEL_SIZE=${SLURM_NNODES}
export TENSOR_PARALLEL_SIZE=4

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


echo "Check environment"
echo "- HEAD NODE: ${head_node}"
echo "- IP ADDRESS: ${head_node_ip}"
echo "- SLURM JOB NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "- nodes: ${nodes}"
echo "- PIPELINE PARALLEL SIZE: ${PIPELINE_PARALLEL_SIZE}"
echo "- TENSOR PARALLEL SIZE: ${TENSOR_PARALLEL_SIZE}"

echo "Start Execution"
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

# --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" 
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --block &
    sleep 5
done


vllm serve deepseek-ai/DeepSeek-R1 \
--tensor-parallel-size $TENSOR_PARALLEL_SIZE \
--pipeline-parallel-size $PIPELINE_PARALLEL_SIZE \
--trust-remote-code \
--distributed-executor-backend ray
