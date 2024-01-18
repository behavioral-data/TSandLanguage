#!/bin/bash

# # For running category question generation
# python qa_cat.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1 1000 &
# python qa_cat.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1000 2000 &
# python qa_cat_az_k1.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 2000 3000 &
# python qa_cat_az_k1.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 3000 4000 &
# python qa_cat_az_k1.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 4000 5000 &
# python qa_cat_az_k2.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 5000 6000 &
# python qa_cat_az_k2.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 6000 7500 &
# python qa_cat_az_k2.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 7500 -1 &


# # For running time-series question generation
# python neg_ts.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1 1000 &
# python neg_ts.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1000 2000 &
# python neg_ts_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 2000 3000 &
# python neg_ts_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 3000 4000 &
# python neg_ts_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 4000 5000 &
# python neg_ts_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 5000 6000 &
# python neg_ts_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 6000 7500 &
# python neg_ts_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 7500 -1 &

# For evaluating over CAT questions
# python eval_cat_wdump.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1 1000 &
# python eval_cat_wdump.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1000 2000 &
# python eval_cat_wdump_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 2000 3000 &
# python eval_cat_wdump_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 3000 4000 &
# python eval_cat_wdump_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 4000 5000 &
# python eval_cat_wdump_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 5000 6000 &
# python eval_cat_wdump_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 6000 7500 &
# python eval_cat_wdump_az.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 7500 -1 &

# For getting CAT questions
python neg_cat.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1 1000 &
python neg_cat.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 1000 2000 &
python neg_cat_az_k1.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 2000 3000 &
python neg_cat_az_k1.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 3000 4000 &
python neg_cat_az_k1.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 4000 5000 &
python neg_cat_az_k2.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 5000 6000 &
python neg_cat_az_k2.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 6000 7500 &
python neg_cat_az_k2.py /gscratch/bdata/datasets/llms_and_timeseries/v2_8787.jsonl 6000 7500 &

