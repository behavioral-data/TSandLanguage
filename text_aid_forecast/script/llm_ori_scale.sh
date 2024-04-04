#!/bin/bash

# 'TS as Plain Text (GPT-4) wo/ Context'
# Keep original data scale for prediction without context.
python3 ./tsllm/main.py experiment=llm_wo_context model=gpt-4  

# 'TS as Plain Text (GPT-4) w/ all Context'
# Keep original data scale for prediction with all context. 
# The data scale is consistent with the information in the captions within the context.
python3 ./tsllm/main.py experiment=llm_wi_all model=gpt-4     