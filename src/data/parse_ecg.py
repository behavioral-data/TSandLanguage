"""
Takes the mean of the ECG Signal, applies train-test split, and saves
"""
import os

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import read_jsonl, write_jsonl

def parse_item(item):
    ecg = item['ecg']
    ts = []
    for lead in ecg:
        ts.append([float(x) for x in ecg[lead]['readings'].split(',')])
    result = np.mean(ts, axis = 0)
    item["mean_signal"] = result

OUT_PATH = "/gscratch/bdata/mikeam/SensingResearch/data/processed/ecg"
INPUT_PATH = "/gscratch/bdata/vinayak/ts4llm/data/ECG.json"

data = read_jsonl(INPUT_PATH)
for item in data:
    parse_item(item)

train, test = train_test_split(data, test_size=0.33, random_state=42)

write_jsonl(train, os.path.join(OUT_PATH, "train.jsonl"))
write_jsonl(test, os.path.join(OUT_PATH, "test.jsonl"))
write_jsonl(data, os.path.join(OUT_PATH, "all.jsonl"))