#!/bin/bash

#  'LLM-TIME (GPT-4) wo/ Context'
#   Use the LLMTIME method to forecast without the help of context. (LLMTIME : https://github.com/ngruver/llmtime)

python3 ./tsllm/main.py experiment=llmtime_wo_context model=gpt-4     
