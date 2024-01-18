import pdb
import os
import sys
import numpy as np
import pickle
import random
import json
import string
import re
from pathlib import Path

def filter_out(ts_list, model, wts):
	new_list = []
	for ts in ts_list:
		ts_split = ts.split("_")
		if model == ts_split[2].split(".")[0]:
			new_list.append(ts)
	return new_list

def get_results():
	# Data Loading
	folder_loc = "Results/"
	model = sys.argv[1]
	wts = int(sys.argv[2])
	ts_list = filter_out(os.listdir(folder_loc), model, wts)

	correct = 0
	cat_wise = {}
	all_mcqs = 0

	for ts_file in ts_list:
		file_name = folder_loc+ts_file
		dataset = pickle.load(open(file_name, 'rb'))
		description, cat_list, preds, cor_inds = dataset[0], dataset[1], dataset[2], dataset[3]

		if preds == cor_inds:
			correct += 1
		all_mcqs += 1

		# for i in range(len(cat_list)):
		# 	cat = cat_list[i]
		# 	if cat not in cat_wise:
		# 		cat_wise[cat] = [0, 0]
		# 	if preds[i] == cor_inds[i]:
		# 		cat_wise[cat][0] += 1
		# 		correct += 1
		# 	cat_wise[cat][1] += 1
		# 	all_mcqs += 1
	print(str(correct)+"/"+str(all_mcqs)+" -- "+str(correct/all_mcqs))
	# print(cat_wise)

if __name__=="__main__": 
	get_results() 
