#!/bin/bash
# Divide into smaller dumps
python divide_ts.py

wait

# Get Codes
python get_code_dumps_2TS.py 0 500 &
python get_code_dumps_2TS.py 500 1000 &
python get_code_dumps_2TS.py 1000 1500 &
python get_code_dumps_2TS.py 1500 2000 &
python get_code_dumps_2TS.py 2000 2500 &
python get_code_dumps_2TS.py 2500 3000 &
python get_code_dumps_2TS.py 3000 3500 &
python get_code_dumps_2TS.py 3500 4000 &
python get_code_dumps_2TS.py 4000 4500 &
python get_code_dumps_2TS.py 4500 5000 &
python get_code_dumps_2TS.py 5000 5500 &
python get_code_dumps_2TS.py 5500 6000 &
python get_code_dumps_2TS.py 6000 6500 &
python get_code_dumps_2TS.py 6500 7000 &
python get_code_dumps_2TS.py 7000 7500 &
python get_code_dumps_2TS.py 7500 8000 &
python get_code_dumps_2TS.py 8000 -1 &

wait

# # Get Questions
python get_ques_dump_2TS.py 0 500 &
python get_ques_dump_2TS.py 500 1000 &
python get_ques_dump_2TS.py 1000 1500 &
python get_ques_dump_2TS.py 1500 2000 &
python get_ques_dump_2TS.py 2000 2500 &
python get_ques_dump_2TS.py 2500 3000 &
python get_ques_dump_2TS.py 3000 3500 &
python get_ques_dump_2TS.py 3500 4000 &
python get_ques_dump_2TS.py 4000 4500 &
python get_ques_dump_2TS.py 4500 5000 &
python get_ques_dump_2TS.py 5000 5500 &
python get_ques_dump_2TS.py 5500 6000 &
python get_ques_dump_2TS.py 6000 6500 &
python get_ques_dump_2TS.py 6500 7000 &
python get_ques_dump_2TS.py 7000 7500 &
python get_ques_dump_2TS.py 7500 8000 &
python get_ques_dump_2TS.py 8000 -1 &

wait

python mcq_map_2TS.py

wait

# Get Results
python gpt_all_2TS.py 0 500 gpt3 0 &
python gpt_all_2TS.py 500 1000 gpt3 0 &
python gpt_all_2TS.py 1000 1500 gpt3 0 &
python gpt_all_2TS.py 1500 2000 gpt3 0 &
python gpt_all_2TS.py 2000 2500 gpt3 0 &
python gpt_all_2TS.py 2500 3000 gpt3 0 &
python gpt_all_2TS.py 3000 3500 gpt3 0 &
python gpt_all_2TS.py 3500 4000 gpt3 0 &
python gpt_all_2TS.py 4000 4500 gpt3 0 &
python gpt_all_2TS.py 4500 5000 gpt3 0 &
python gpt_all_2TS.py 5000 5500 gpt3 0 &
python gpt_all_2TS.py 5500 6000 gpt3 0 &
python gpt_all_2TS.py 6000 6500 gpt3 0 &
python gpt_all_2TS.py 6500 7000 gpt3 0 &
python gpt_all_2TS.py 7000 7500 gpt3 0 &
python gpt_all_2TS.py 7500 8000 gpt3 0 &
python gpt_all_2TS.py 8000 -1 gpt3 0 &
