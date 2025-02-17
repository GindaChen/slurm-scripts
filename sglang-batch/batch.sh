#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:06:00
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=sglang_server
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module load conda
module load cudatoolkit/12.4
conda activate sglang
# python -m sglang.launch_server --model-path "Qwen/Qwen2.5-7B-Instruct" --tp 2

# Create logs directory structure
LOG_DIR="$(realpath .)/logs/slurm_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

JOBLOG="${LOG_DIR}/runner.log"
ENVLOG="${LOG_DIR}/env.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
nnodes=${#nodes_array[@]}
head_node=${nodes_array[0]}
model="Qwen/Qwen2.5-7B-Instruct"
tp_size=4
pp_size=$nnodes

# Get the IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"

echo "Environment variables:" &>> ${ENVLOG}
env >> ${ENVLOG}
echo "----------------------------------------" &>> ${ENVLOG}
echo "nodes: $nodes" &>> ${ENVLOG}
echo "nodes_array: ${nodes_array[@]}" &>> ${ENVLOG}
echo "head_node: $head_node" &>> ${ENVLOG}
echo "head_node_ip: $head_node_ip" &>> ${ENVLOG}
echo "model: $model" &>> ${ENVLOG}
echo "tp_size: $tp_size" &>> ${ENVLOG}
echo "pp_size: $pp_size" &>> ${ENVLOG}
echo "nnodes: $nnodes" &>> ${ENVLOG}
echo "NCCL_INIT_ADDR: $NCCL_INIT_ADDR" &>> ${ENVLOG}
echo "----------------------------------------" &>> ${ENVLOG}


# Handle potential space-separated IP addresses (IPv4/IPv6)
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<< "$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}  # Use IPv4 if available
    else
        head_node_ip=${ADDR[0]}
    fi
fi

export OUTLINES_CACHE_DIR="/tmp/node_0_cache"


# SLURM filename patterns:
# %j - Job ID
# %x - Job name (from --job-name or script name)
# %N - Node name where output written
# %n - Task number for the job step
# %u - User name
echo "Starting server on node 0"
srun --ntasks=1 --nodes=1 --exclusive --output="${LOG_DIR}/node0.out" --error="${LOG_DIR}/node0.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --host 0.0.0.0 --port 9876 \
    --dist-init-addr ${head_node_ip}:5000 \
    --tp "$tp_size" \
    --dp-size 2 \
    --nnodes "$nnodes" \
    --node-rank 0 &

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 1; i < worker_num; i++)); do
node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"
    
    srun --ntasks=1 --nodes=1 --exclusive --output="${LOG_DIR}/node${i}.out" \
    --error="${LOG_DIR}/node${i}.err" \
    python3 -m sglang.launch_server \
    --host 0.0.0.0 --port 9887 \
    --dist-init-addr ${head_node_ip}:5000 \
    --model-path "$model" \
    --tp "$tp_size" \
    --dp-size 2 \
    --nnodes "$nnodes" \
    --node-rank "$i" &
done
# --nccl-init-addr "$NCCL_INIT_ADDR" \


while ! nc -z localhost 30000; do
    sleep 1
    echo "[INFO] Waiting for localhost:30000 to accept connections"
done

echo "[INFO] localhost:30000 is ready to accept connections"


response=$(curl -s -X POST http://127.0.0.1:30000/v1/chat/completions \
-H "Authorization: Bearer None" \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "List 3 countries and their capitals."
    }
  ],
  "temperature": 0,
  "max_tokens": 64
}')

echo "[INFO] Response from server:"
echo "$response" | tee "${LOG_DIR}/response.log"

echo "[INFO] Job ${SLURM_JOB_ID} completed"
