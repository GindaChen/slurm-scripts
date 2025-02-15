#!/bin/bash

MODEL="facebook/opt-125m"

mkdir -p output_tmp

python run_batch_server.py \
--disable-log-requests \
--multi-step-stream-outputs \
--num-scheduler-steps 32 \
--enable-prefix-caching \
--model $MODEL \
-i examples/input.txt \
-o output_tmp/output.jsonl \
--output-tmp-dir output_tmp/ \
--custome-request-script examples/template.py \
--enforce-eager