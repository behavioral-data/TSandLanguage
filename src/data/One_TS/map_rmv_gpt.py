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

def clean_str(sample):
	delimits = string.punctuation.replace('?','').replace('.','')
	return sample.translate(str.maketrans('', '', delimits)).strip()

def get_components(cur_line):
	ques = clean_str(cur_line.split('question')[1].split('answer')[0])
	ans = clean_str(cur_line.split('answer')[1].split('incorrect')[0])
	opt1 = clean_str(cur_line.split('incorrect answer 1')[1].split('incorrect')[0])
	opt2 = clean_str(cur_line.split('incorrect answer 2')[1].split('incorrect')[0])
	opt3 = clean_str(cur_line.split('incorrect answer 3')[1].split('}')[0])

	options = [ans, opt1, opt2, opt3]
	cor = options[0]
	np.random.shuffle(options)
	cor_ind = options.index(cor)
	return ques, options, cor_ind

def check_type(cur_line, qtype):
	if clean_str(cur_line.split(":")[0]) == 'Type':
		qtype = clean_str(cur_line.split(":")[1])
		if qtype == 'DS':
			return 'description'
		else:
			return 'time-series'
	return qtype

def prep_mcq_ts(ques_dump):
	ques_dump = ques_dump.split("\n")
	loaded_file = []
	no_val = 0
	qtype = ''
	for cur_line in ques_dump:
		if len(cur_line.strip()) > 5:
			qtype = check_type(cur_line, qtype)
			if qtype == 'description':
				if 'code' in cur_line or 'np' in cur_line:
					continue
			try:
				ques, options, cor_ind = get_components(cur_line)
				loaded_file.append([qtype, ques, options, cor_ind])
			except:
				no_val += 1
	return loaded_file

def prep_mcq_cat(ques_dump):
	# This is saved in the format of list [mcq_list, preds, inc_list]
	if len(ques_dump[0]) == 0:
		return []
	inc_ans = []
	parsed = set()

	for i in range(len(ques_dump[2])):
		inc_ans.append(ques_dump[0][i])
		parsed.add(i)

	while len(inc_ans) <= 2:
		# print(len(ques_dump[0]))
		index = np.random.randint(len(ques_dump[0]))
		if index in parsed:
			continue
		inc_ans.append(ques_dump[0][index])
	return inc_ans

def absent_or_blank(save_file):
	my_file = Path(save_file)
	if my_file.is_file():
		return os.stat(save_file).st_size == 0
	return True

def data_loader_mike(dataset_loc):
	dataset = []
	with open(dataset_loc) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset
	
def main():
	# Data Loading
	dataset_loc = sys.argv[1]
	dataset = data_loader_mike(dataset_loc)
	new_data_list = []
	
	for i in range(len(dataset)):
		ts_dump = 'MCQ/Dumps/'+str(i)+'_TS.p'
		llm_cor = 'MCQ/GPT_Test/'+str(i)+'_CAT.p'

		try_cat = 1
		try_ts = 1
		count = 0

		# Check if the file exists or not. We will only use it if it exists.
		if absent_or_blank(llm_cor) == True:
			try_cat = 0

		if absent_or_blank(ts_dump) == True:
			try_ts = 0

		if try_ts == 1:
			mcq_list_ts = prep_mcq_ts(pickle.load(open(ts_dump, 'rb')))
			for mcq in mcq_list_ts:
				dataset[i]['category'] = mcq[0]
				dataset[i]['question'] = mcq[1]
				dataset[i]['options'] = mcq[2]
				dataset[i]['answer_index'] = int(mcq[3])
				dataset[i]['ts_qid'] = count
				new_data_list.append(dataset[i].copy())
				count += 1
			
		if try_cat == 1:
			mcq_list_cat = prep_mcq_cat(pickle.load(open(llm_cor, 'rb')))
			for mcq in mcq_list_cat:
				dataset[i]['category'] = mcq[0]
				dataset[i]['question'] = mcq[1]
				dataset[i]['options'] = mcq[2]
				dataset[i]['answer_index'] = int(mcq[3])
				dataset[i]['ts_qid'] = count
				new_data_list.append(dataset[i].copy())
				count += 1

		print("Reached: "+str(i))

	with open('MCQ/v2_8787_MCQ.json', 'w') as f:
		for d in new_data_list:
			json.dump(d, f)
			f.write('\n')

if __name__=="__main__": 
	main() 
