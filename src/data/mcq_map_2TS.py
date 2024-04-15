import pdb
import os
import sys
import numpy as np
import pickle
import random
import time
import json
import string
import re
import base64
from pathlib import Path
import matplotlib.pyplot as plt
from textwrap import wrap

def plotter_title(original, new_series, question, app):
	x = np.arange(len(original))
	plt.plot(x, original, "-b", label="O")
	plt.title(question, wrap=True)
	plt.tick_params(axis='both', labelsize=10)
	plt.savefig("Plots/HQ_T/"+str(app)+'_O.png')
	plt.close()

	x = np.arange(len(new_series))
	plt.plot(x, new_series, "-r", label="C")
	plt.title(question, wrap=True)
	plt.tick_params(axis='both', labelsize=10)
	plt.savefig("Plots/HQ_T/"+str(app)+'_C.png')
	plt.close()

def plotter(original, new_series, app):
	x = np.arange(len(original))
	plt.plot(x, original, "-b", label="O")
	plt.tick_params(axis='both', labelsize=10)
	plt.savefig("Plots/HQ/"+str(app)+'_O.png')
	plt.close()

	x = np.arange(len(new_series))
	plt.plot(x, new_series, "-r", label="C")
	plt.tick_params(axis='both', labelsize=10)
	plt.savefig("Plots/HQ/"+str(app)+'_C.png')
	plt.close()

def load_data(file):
	dataset = []
	with open(file) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset

dataset = load_data('v2_8787.jsonl')
ques_list = os.listdir("Dumps/")
q_dict = {}
new_data_list = [] 

for i in range(len(ques_list)):
	if ques_list[i].split('_')[-1]=='Ques.p':
		uuid = ques_list[i].split('_')[0]
		if uuid not in q_dict:
			q_dict[uuid] = []
		q_dict[uuid].append(ques_list[i])

for i in range(len(dataset)):
	uuid = dataset[i]['uuid']
	if uuid not in q_dict: 
		continue
	dumps_list = q_dict[uuid]
	for d in dumps_list:
		code_file = d.replace('Ques', 'Code')
		ques_dump = pickle.load(open('Dumps/'+d, 'rb'))
		code_dump = pickle.load(open('Dumps/'+code_file, 'rb'))
		count = 0
		for ques in ques_dump:
			if isinstance(code_dump[0], list) == True:
				dataset[i]['new_series'] = code_dump[0]
			else:
				dataset[i]['new_series'] = [c for c in code_dump[0].astype(np.float64)]
			dataset[i]['new_generator'] = code_dump[1]
			dataset[i]['counterfactual'] = code_dump[3]
			dataset[i]['category'] = ques[0]
			dataset[i]['question'] = ques[1]
			dataset[i]['options'] = ques[2]
			dataset[i]['answer_index'] = int(ques[3])
			dataset[i]['ts_qid'] = d[:-7]
			dataset[i]['ts_count'] = count
			new_data_list.append(dataset[i].copy())
			count += 1
		plotter(dataset[i]['series'], dataset[i]['new_series'], d[:-7])
		plotter_title(dataset[i]['series'], dataset[i]['new_series'], code_dump[3], d[:-7])
		print(i, len(new_data_list))

with open('Two_TS_CF_MCQS.json', 'w') as f:
	for d in new_data_list:
		json.dump(d, f)
		f.write('\n')
