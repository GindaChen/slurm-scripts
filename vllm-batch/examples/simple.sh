#!/bin/bash

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

python run_batch_server.py \
--disable-log-requests \
--multi-step-stream-outputs \
--num-scheduler-steps 32 \
--enable-prefix-caching \
--model $MODEL \
-i examples/input.jsonl
