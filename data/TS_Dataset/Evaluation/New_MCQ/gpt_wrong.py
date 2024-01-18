import pdb
import os
import sys
import numpy as np
import pickle
import random
import time
import openai
import backoff
import json
import string
import re
from pathlib import Path

def print_if_needed(inc_list):
	ret_str = ''
	for i in range(len(inc_list)):
		ind_str = ''
		ind_str += 'Question #'+str(i+1)+': '+inc_list[i]['question']+'\n'
		for j in range(len(inc_list[i]['options'])):
			ind_str += 'Option '+opt_2_char(j)+': '+inc_list[i]['options'][j]+'\n'
		ind_str += 'Correct Option: ' + opt_2_char(inc_list[i]['label'])+'\n'
		ind_str += 'GPT Selection: ' + opt_2_char(inc_list[i]['gpt_pred'])
		ret_str = ret_str+'\n\n'+ind_str
	print(ret_str)
		

def filter_out(ts_list, model, wts):
	new_list = []
	for ts in ts_list:
		ts_split = ts.split("_")
		if model == ts_split[2] and wts == int(ts_split[3][0]):
			new_list.append(ts)
	return new_list

# def get_incorrect(dval):

def char_2_opt(opt):
	opt_chars = {'A':0, 'B':1, 'C':2, 'D':3}
	return opt_chars[opt]

def opt_2_char(opt):
	opt_chars = ['A', 'B', 'C', 'D']
	return opt_chars[opt]

def load_data(file):
	dataset = []
	with open(file) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset

def absent_or_blank(save_file):
	my_file = Path(save_file)
	if my_file.is_file():
		return os.stat(save_file).st_size == 0
	return True

def get_file_with_ind(ts_list, ind):
	for ts in ts_list:
		if ts.split('_')[0] == str(ind):
			return ts

def get_incorrect():
	# Data Loading
	res_loc = "Results/"
	data_loc = "TS_Questions/"
	model = sys.argv[1]
	res_list = filter_out(os.listdir(res_loc), model, 0)
	ts_list = os.listdir(data_loc)
	inc_list = []

	for i in range(len(res_list)):
		file = res_list[i]
		
		if absent_or_blank(res_loc+file) == True:
			continue
		
		results = pickle.load(open(res_loc+file, 'rb'))
		mcq_list, cat_list, preds, cor_inds = results[0], results[1], results[2], results[3]

		# Load file that has dataset.
		datafile = get_file_with_ind(ts_list, int(file.split('_')[0]))
		dataset = load_data(data_loc+datafile)

		for i in range(len(preds)):
			if preds[i] != cor_inds[i]:
				dataset[i]['gpt_pred'] = char_2_opt(preds[i])
				inc_list.append(dataset[i].copy())

	print_if_needed(inc_list)

	# with open('GPT_Incorrect.json', 'w') as f:
	# 	for d in inc_list:
	# 		json.dump(d, f)
	# 		f.write('\n')

if __name__=="__main__": 
	get_incorrect() 
