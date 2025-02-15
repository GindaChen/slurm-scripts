#!/bin/bash

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

python run_batch_server.py \
--disable-log-requests \
--multi-step-stream-outputs \
--num-scheduler-steps 32 \
--enable-prefix-caching \
--model $MODEL \
-i examples/input.txt \
--custome-request-script examples/template.py