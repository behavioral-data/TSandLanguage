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
from pathlib import Path

'''
In this code, we will generate negative answers for every question. 
	1. _CAT
'''

# GPT Functions ---------------->
def send_to_gpt3(prompt):
	openai.api_base = "https://ts-language-oai.openai.azure.com/"
	openai.api_key = os.environ["OPENAI_API_KEY_AZURE"]
	openai.api_version = "2023-07-01-preview"
	openai.api_type = "azure"
	deployment = "gpt3-16k"

	MAX_RETRIES = 10
	current_tries = 1
	while current_tries <= MAX_RETRIES:
		try:
			response = openai.ChatCompletion.create(engine=deployment,messages=[{"role": "user", "content": prompt}],max_tokens=4000)
			break
		except Exception as e:
			print('openai retrying, error:', str(e))
			time.sleep(10)
			current_tries += 1

	response_text = response['choices'][0]['message']['content']
	return response_text

def send_to_gpt4(prompt):
	openai.api_base = "https://ts-language-oai.openai.azure.com/"
	openai.api_key = os.environ["OPENAI_API_KEY_AZURE"]
	openai.api_version = "2023-09-01-preview"
	openai.api_type = "azure"
	deployment = "gpt-4"

	MAX_RETRIES = 5
	current_tries = 1
	while current_tries <= MAX_RETRIES:
		try:
			response = openai.ChatCompletion.create(deployment_id=deployment,model="gpt-4",messages=[{"role": "user", "content": prompt}],max_tokens=2000)
			break
		except Exception as e:
			print('openai retrying, error:', str(e))
			time.sleep(10)
			current_tries += 1

	response_text = response['choices'][0]['message']['content']
	return response_text

# End GPT Functions ---------------->

def clean_fact_questions(fact_file):
	combined = '\n'.join(fact_file)
	fact_file = []
	init_text = "You have been given a set of question-answer pairs related to a time series. If you encounter any mention of any description, Python code or any 'characteristic' in either the question or the answer, replace the text so that it appears as if it was generated in the absence of any description, code or characteristics. Only replace the text that refers to any description, code or 'characteristic'; keep all other information as is. After making the replacements, return the entire text in the exact same format. Do not generate any additional text.\n"
	gpt_txt = init_text+combined
	gpt_res = send_to_gpt4(gpt_txt).split('\n')
	fact_file = [ques for ques in gpt_res if len(ques)>5]
	return fact_file

def get_qa_gpt_neg(prompt_str):
	init_text = "Given a description of a time-series and a set of question-answer pairs, create three incorrect answer options for each question. Your incorrect answers should have similar lengths compared to the correct answers. The input format is: '{'category':'', 'question':'', 'answer':''}'. In the output, you should copy the question and answers from the input and provide incorrect options in the following format: '{'category':'', 'question':'', 'answer':'', 'incorrect answer 1':'', 'incorrect answer 2':'', 'incorrect answer 3':''}\n'. Each new question should start on a new line. Do not separate question, its answer and options into different lines. Ensure that each question, its corresponding answer and incorrect answers are presented on the same line. Do not use any double quotations within the text. Try to make the negative solutions sound similar to the correct answer but diffferent from other negative options. If, in any question or answer, you see the mention of any Python code or any 'characteristic', then replace it with the term 'time-series'. Avoid the use of contractions in all kinds of notations. Instead, use the full forms for greater clarity. If there exists any contraction in the question or answer, then replace it with the full-form. Do not generate any additional text.\n"
	gpt_txt = init_text+prompt_str
	gpt_res = send_to_gpt3(gpt_txt)
	return [gpt_res, gpt_txt]

def prep_neg_prompt(desc, chars, qa):
	it = 1
	qa_str = ''
	for q in qa:
		qa_str += 'Question '+str(it)+' '+q+'\n'
		it += 1

	prompt_str = 'Description: '+desc+'\n'+'Characteristics: '+chars+'\n'+'Questions: '+qa_str
	return prompt_str

def load_file(index):
	loaded_file = []
	fact_file = []
	add = "MCQ/CAT/"+str(index)+"_CAT.txt"
	with open(add, 'r') as f:
		for line in f:
			if len(line.strip()) > 5:
				if "'fact'" in line:
					fact_file.append(line)
				else:
					loaded_file.append(line)
	fact_file_2 = clean_fact_questions(fact_file)
	loaded_file.extend(fact_file)
	return loaded_file

def neg_options(dval, index):
	desc = dval['description'].replace('\n', '').strip()
	desc_s = dval['description_short'].replace('\n', '').strip()
	desc_t = dval['description_tiny'].replace('\n', '').strip()
	chars = dval['characteristics'].replace('\n', '').strip()
	code = dval['generator'].strip()
	meta = dval['metadata']
	mcq_list = []

	try:
		output = ''
		loaded_file = load_file(index)
		for i in range(0, len(loaded_file), 5):
			# desc + chars --> GPT
			prompt_str = prep_neg_prompt(desc, chars, loaded_file[i:(i+5)])
			output += get_qa_gpt_neg(prompt_str)[0]+"\n"

		# Dumping the output for future references
		pickle.dump(output, open('MCQ/Dumps/'+str(index)+'_CAT.p', 'wb'))
		return output
	except:
		return []
	
	return mcq_list

def data_loader_mike(dataset_loc):
	dataset = []
	with open(dataset_loc) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset
	
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

def main():
	# Data Loading
	dataset_loc = sys.argv[1]
	dataset = data_loader_mike(dataset_loc)
	to_start = int(sys.argv[2])
	to_end = int(sys.argv[3])

	if to_end == -1:
		to_end = len(dataset)

	new_data_list = []
	
	for i in range(to_start, to_end):
		save_file = 'MCQ/Dumps/'+str(i)+'_CAT.p'
		if absent_or_blank(save_file) == False:
			print("Reached: "+str(i))
			continue

		dval = dataset[i]
		mcq_list = neg_options(dval, i)

		if mcq_list == []:
			print("Check for: "+str(i))
		else:
			print("Reached: "+str(i))

if __name__=="__main__": 
	main() 
