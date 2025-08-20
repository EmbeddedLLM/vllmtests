#!/bin/bash

lm_eval \
--model local-completions \
--tasks gsm8k \
--model_args model=Qwen/Qwen3-235B-A22B-FP8,base_url=http://127.0.0.1:6789/v1/completions \
--batch_size 100 \
> lmeval_server-Qwen_Qwen3-235B-A22B-FP8-aiter-v1-fp8-cudagraph_FULL.log 2>&1