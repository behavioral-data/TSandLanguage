#!/bin/bash
# Use the LLMTIME method to forecast with the help of various captions. (LLMTIME : https://github.com/ngruver/llmtime)

# 'LLM-TIME (GPT-4) w/ Ca',
python3 ./tsllm/main.py experiment=llmtime_wi_Ca model=gpt-4    

# 'LLM-TIME (GPT-4) w/ Ch',
python3 ./tsllm/main.py experiment=llmtime_wi_Ch model=gpt-4    

# 'LLM-TIME (GPT-4) w/ ChMe'
python3 ./tsllm/main.py experiment=llmtime_wi_ChMe model=gpt-4     

# 'LLM-TIME (GPT-4) w/ CaMe',
python3 ./tsllm/main.py experiment=llmtime_wi_CaMe model=gpt-4      

# Note that : Caption is 'description' filed in dataset. Caption (Ca) , Characteristics (Ch), Metadata (Me)