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
from pathlib import Path

'''
Only for TS and DS specific questions.
'''

TS_tokens = 1500

# GPT Functions ---------------->
openai.api_key = "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"

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def send_to_gpt3(sendpromt):
	completion = openai.ChatCompletion.create(
		model="gpt-3.5-turbo-1106",
		max_tokens=3000,
		messages=[{"role": "user", "content": sendpromt}])
	return completion

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def send_to_gpt4(sendpromt):
	completion = openai.ChatCompletion.create(
		model="gpt-4-1106-preview",
		max_tokens=4000,
		messages=[{"role": "user", "content": sendpromt}])
	return completion

def verbalizer(gpt_output):
	message = gpt_output['choices'][0]['message']['content']
	return message
# End GPT Functions ---------------->

def get_ts_qa_gpt_4(prompt_str_ts):
	init_text = "Given a description of a time-series, a set of sentences describing its characteristics, and a python code segment that generates this time-series. You have to create ten question-answer pairs. Formulate questions that someone might naturally ask about the time-series that need numerical values as answers. Do not ask any questions directly related to the description or the python code. The aim is to make the questions appear as if they have been organically posed about the values within the time-series. Scale your answers as per the maximum and minimum values of the time-series as given. For a numeric answer, you can provide an approximate value. Just be sure to indicate explicitly that the response is an approximation for transparency and clarity. Provide the questions and answers in the format: '{'question':'', 'answer':''}'. Ensure that each question and its corresponding answer are presented on the same line, with each new question starting on a new line for a clear and organized format. Do not generate any additional text.\n"
	gpt_txt = init_text+prompt_str_ts
	gpt4_res = verbalizer(send_to_gpt4(gpt_txt))
	return [gpt4_res, gpt_txt]

def get_desc_qa_gpt_3(prompt_str):
	init_text = "Given a description of a time-series, a set of sentences describing its characteristics, create ten question-answer pairs. Formulate questions using the description and the characteristics and should look like what individuals might naturally ask about the time-series. Provide the questions and answers in the following exact format: '{'question':'', 'answer':''}'. Ensure that each question and its corresponding answer are presented on the same line, with each new question starting on a new line for a clear and organized format. Do not generate any additional text.\n"
	gpt_txt = init_text+prompt_str
	gpt3_res = verbalizer(send_to_gpt3(gpt_txt))
	return [gpt3_res, gpt_txt]

def prep_ts_prompt(desc, ts, code, chars=''):
	if chars == '':
		prompt_str = 'Description:'+desc+'\n'+'Code:'+code+'\n'+'Time Series:'+ts
	else:
		prompt_str = 'Description:'+desc+'\n'+'Characteristics:'+chars+'\n'+'Code:'+code+'\n'+'Time Series:'+ts
	return prompt_str

def prep_non_ts_prompt(desc, code, chars=''):
	if chars == '':
		prompt_str = 'Description:'+desc+'\n'+'Code:'+code
	else:
		prompt_str = 'Description:'+desc+'\n'+'Characteristics:'+chars+'\n'+'Code:'+code
	return prompt_str

def get_max_min(ts):
	ret_str = 'The maximum value of the time-series is: '+str(format(max(ts), '.2f'))+' and the minimum value is:'+str(format(min(ts), '.2f'))+'.'
	return ret_str

# Not required if you are passing the complete time-series. 
def fix_ts(ts, type_ts = 'uniform'):
	global TS_tokens
	TS_tokens = min(TS_tokens, len(ts))
	if type_ts == 'uniform':
		indices = np.sort(np.random.choice(len(ts), TS_tokens, replace=False))
		ts = [ts[i] for i in indices]

	if type_ts == 'first':
		ts = ts[:TS_tokens]

	return ts

def ts2str(ts):
	ts_arr = []
	for val in ts:
		# LLM-TIME states that it has only 2 digit precision.
		val = format(val, '.2f')
		ts_word = ''
		for j in val:
			if j == '.':
				continue
			ts_word += j+" "
		ts_word = ts_word.strip() 
		ts_arr.append(ts_word)
	return ' , '.join(ts_arr).strip()

def absent_or_blank(save_file):
	my_file = Path(save_file)

	# Check if file exists
	if my_file.is_file():
		if os.stat(save_file).st_size == 0:
			# File exists and is blank.
			return True
		else:
			# File exists and is full.
			return False
	# File does not exist.
	return True

# We use GPT to get TS-specific Question and Answer Pairs.
def get_ts_ques(dval, index):
	filename = "./MCQ/Files/"
	save_file = filename+str(index)+"_TS.txt"
	# To check if the file exists and is blank.
	if absent_or_blank(save_file) == False:
		return

	desc = dval['description'].replace('\n', '').strip()
	desc_s = dval['description_short'].replace('\n', '').strip()
	desc_t = dval['description_tiny'].replace('\n', '').strip()
	chars = dval['characteristics'].replace('\n', '').strip()
	code = dval['generator'].strip()
	meta = dval['metadata']
	ts = dval['series']
	ts_str = ts2str(ts)

	# Create directory:
	filename = "./MCQ/Files/"
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	f1 = open(filename+str(index)+"_TS.txt", "w")
	f2 = open(filename+str(index)+"_Prompts.txt", "a")

	# desc + ts_str + code --> GPT-4
	desc += get_max_min(ts)
	prompt_str = prep_ts_prompt(desc, ts_str, code)
	output = get_ts_qa_gpt_4(prompt_str)
	res = output[0]
	prompt_res = output[1]

	# If output length is small, then ignore.
	if len(res) < 5:
		return

	f1.write(res+'\n')
	f2.write(prompt_res+'\n')
	f1.close()
	f2.close()

# We use GPT to get description related questions.
def get_desc_ques(dval, index):
	filename = "./MCQ/Files/"
	save_file = filename+str(index)+"_DS.txt"
	# To check if the file exists and is blank.
	if absent_or_blank(save_file) == False:
		return

	desc = dval['description'].replace('\n', '').strip()
	chars = dval['characteristics'].replace('\n', '').strip()
	code = dval['generator'].strip()
	meta = dval['metadata']
	ts = dval['series']
	ts_str = ts2str(ts)

	# Create directory:
	filename = "./MCQ/Files/"
	f1 = open(filename+str(index)+"_DS.txt", "w")
	f2 = open(filename+str(index)+"_Prompts.txt", "a")

	# desc + code + chars --> GPT-3
	prompt_str = prep_non_ts_prompt(desc, code, chars)
	output = get_desc_qa_gpt_3(prompt_str)
	res = output[0]
	prompt_res = output[1]

	# MCQ
	if len(res) < 5:
		return

	f1.write(res+'\n')
	f2.write(prompt_res+'\n')
	f1.close()
	f2.close()

def data_loader_mike(dataset_loc):
	dataset = []
	with open(dataset_loc) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset

def main():
	# Data Loading
	dataset_loc = sys.argv[1]
	to_start = int(sys.argv[2])
	dataset = data_loader_mike(dataset_loc)
	for i in range(to_start, len(dataset)):
		try:
			dval = dataset[i]
			get_ts_ques(dval, i)
			get_desc_ques(dval, i)
			print("Reached: "+str(i))
		except:
			print("An exception occurred")

if __name__=="__main__": 
	main() 
