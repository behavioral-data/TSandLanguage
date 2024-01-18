#!/bin/bash

# For 758
# Run All
python gpt_ts_fact.py 0 200 gpt3 1 &
python gpt_ts_fact.py 200 400 gpt3 1 &
python gpt_ts_fact.py 400 600 gpt3 1 &
python gpt_ts_fact.py 600 -1 gpt3 1 &

python gpt_ts_fact.py 0 200 gpt4 1 &
python gpt_ts_fact.py 200 400 gpt4 1 &
python gpt_ts_fact.py 400 600 gpt4 1 &
python gpt_ts_fact.py 600 -1 gpt4 1 &

python gpt_ts_fact.py 0 200 gpt4 0 &
python gpt_ts_fact.py 200 400 gpt4 0 &
python gpt_ts_fact.py 400 600 gpt4 0 &
python gpt_ts_fact.py 600 -1 gpt4 0 &

# python gpt_ts_fact.py 0 200 gptv 0 &
# python gpt_ts_fact.py 200 400 gptv 0 &
# python gpt_ts_fact.py 400 600 gptv 0 &
# python gpt_ts_fact.py 600 -1 gptv 0 &

