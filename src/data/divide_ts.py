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

'''
To simultaneously send multiple prompts to the OpenAI server, 
this code splits the large dataset into smaller files. 
Later, these files can be loaded based on the indices, 
using a script.
'''

def write_one_line_json(data_list, filename):
	with open(filename, 'w') as f:
		json.dump(data_list, f)
		f.write('\n')

def data_loader_mike(dataset_loc):
	dataset = []
	with open(dataset_loc) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset
	
def main():
	# Data Loading
	dataset_loc = 'v2_8787.jsonl'
	dataset = data_loader_mike(dataset_loc)

	for i in range(len(dataset)):
		save_file = 'Splits/'+str(i)
		write_one_line_json(dataset[i], save_file+'.json')

if __name__=="__main__": 
	main() 
