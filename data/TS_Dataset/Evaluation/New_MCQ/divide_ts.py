import pdb
import os
import sys
import numpy as np
import pickle
import random
import time
import json
import string
from pathlib import Path
import re


def index_dict_uuid(dataset):
	uuid_index = {}
	for i in range(len(dataset)):
		uuid = dataset[i]['uuid']
		if uuid not in uuid_index:
			uuid_index[uuid] = []
		uuid_index[uuid].append(i)
	return uuid_index


def write_json(data_list, filename):
	with open(filename, 'w') as f:
		for d in data_list:
			json.dump(d, f)
			f.write('\n')


def data_loader_mike(dataset_loc):
	dataset = []
	with open(dataset_loc) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset
	

def main():
	# Data Loading
	dataset_loc = 'mike_qa_processed.jsonl'
	dataset = data_loader_mike(dataset_loc)
	uuid_index = index_dict_uuid(dataset)
	ind_keys = list(uuid_index.keys())
	new_data_list = []
	parsed = set()
	count = 0

	for k,v in uuid_index.items():
		save_file = 'TS_Questions/'+str(count)+"_"+k
		count += 1
		temp_list = []

		for j in v:
			new_data_list.append(dataset[j].copy())
			temp_list.append(dataset[j].copy())

		write_json(temp_list, save_file+'.json')
	# write_json(new_data_list, 'New_Mike_MCQ.json')


if __name__=="__main__": 
	main() 
