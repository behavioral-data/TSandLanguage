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
Types of Questions:
	1. Counterfactual Reasoning
	2. Explanation
	3. Argumentation
	4. Analogical Reasoning
	5. Fact Checking
'''

# GPT Functions ---------------->
def send_to_gpt4(prompt):
	openai.api_base = "https://ts-language-oai.openai.azure.com/"
	openai.api_key = os.environ["OPENAI_API_KEY_AZURE_2"]
	openai.api_version = "2023-09-01-preview"
	openai.api_type = "azure"
	deployment = "gpt-4"

	MAX_RETRIES = 10
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

def get_cat_qa_gpt_4(prompt_str):
	ex_types = ['counterfactual', 'explanation', 'argumentation', 'analogical', 'fact']
	final_res = ''
	final_txt = ''
	for et in ex_types:
		init_text = "Given a description of a time-series, a set of sentences describing its characteristics, and a python code segment that generates this time-series. "+give_example(et)+" Create questions and answers that avoid referencing or directly quoting code or the description. Avoid asking questions specifically tied to the description or the Python code. The questions should require an understanding of time-series dynamics for accurate answers. The answers should not mention the description or the code at all. Provide the questions and answers in the following exact format: '{'category':'"+et+"', 'question':'', 'answer':''}'. Ensure that each question and its corresponding answer are presented on the same line, with each new question starting on a new line for a clear and organized format. Do not generate any additional text.\n"

		gpt_txt = init_text+prompt_str
		gpt4_res = send_to_gpt4(gpt_txt)
		final_res = final_res + gpt4_res + "\n\n"
		final_txt = final_txt + init_text + "\n\n"
	return [final_res, final_txt]

def give_example(et):
	if et == 'counterfactual':
		type_prompt = "You have to create five counterfactual question-answer pairs. Counterfactual reasoning questions involve exploring hypothetical scenarios by considering what would have happened if certain events or conditions had been different from what actually occurred. For example, 'What will the time-series look like if some event occured?'. Generate a wide-range of questions."
	elif et == 'explanation':
		type_prompt = "You have to create five question-answer pairs that require examining hypothetical scenarios within a given time-series or description. The answers should articulate the reasons, causes, or underlying mechanisms that would explain the imagined situation. For example, 'What could have led an event to occur based on the time-series?'. Generate a wide-range of questions."
	elif et == 'argumentation':
		type_prompt = "You have to create five question-answer pairs. Formulate questions that present arguments based on the given time-series. The answers to these questions should include compelling arguments based on the time-series, providing well-reasoned points, and supporting evidence where applicable. Demonstrate persuasive language to convey your stance in the answer. For example, 'What should I do a task X based on the dynamics of the time-series?', where X can be any arbitraty task related to the time-series. Generate a wide-range of questions."
	elif et == 'analogical':
		type_prompt = "You have to create five analogical reasoning question-answer pairs. Analogical reasoning questions prompt individuals to identify parallels between different scenarios. The questions should highlight the parallels between various scenarios in analogical reasoning questions. Emphasize how recognizing structural or pattern similarities fosters creative thinking and facilitates the transfer of knowledge, enhancing problem-solving insights."
	elif et == 'fact':
		type_prompt = "You have to create five fact-checking question-answer pairs. Fact-checking questions that explore the accuracy and reliability of the information presented. The answer should indicate whether the point raised in the question is 'True' or 'False' providing supporting evidence when applicable. For example, 'Does the time-series demonstrate a specifc property?'. Generate a wide-range of questions."
	
	return type_prompt

def prep_ts_prompt_new(desc, ts, code, chars=''):
	prompt_str = 'Description:'+desc+'\n'+'Characteristics:'+chars+'\n'+'Code:'+code
	return prompt_str

def prep_ts_prompt(desc, ts, code, chars=''):
	if chars == '':
		prompt_str = 'Description:'+desc+'\n'+'Code:'+code+'\n'+'Time Series:'+ts
	else:
		prompt_str = 'Description:'+desc+'\n'+'Characteristics:'+chars+'\n'+'Code:'+code+'\n'+'Time Series:'+ts
	return prompt_str

def get_max_min(ts):
	ret_str = 'The maximum value of the time-series is: '+str(format(max(ts), '.2f'))+' and the minimum value is:'+str(format(min(ts), '.2f'))+'.'
	return ret_str

def fix_ts(ts, type_ts = 'first'):
	TS_tokens = min(500, len(ts))
	if type_ts == 'uniform':
		indices = np.sort(np.random.choice(len(ts), TS_tokens, replace=False))
		ts = [ts[i] for i in indices]
	if type_ts == 'first':
		ts = ts[:TS_tokens]
	return ts

def ts2str(ts):
	ts = fix_ts(ts)
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
	if my_file.is_file():
		return os.stat(save_file).st_size == 0
	return True

# We use GPT to questions of categories.
def get_cat_ques(dval, index):
	# To check if the file exists and is blank.
	save_file = "./MCQ/CAT/"+str(index)+"_CAT.txt"
	prompt_file = "./MCQ/CAT/"+str(index)+"_Prompts.txt"
	
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
	f1 = open(save_file, "w")
	f2 = open(prompt_file, "w")

	# desc + ts_str + code + chars --> GPT-4
	desc += get_max_min(ts)
	prompt_str = prep_ts_prompt_new(desc_t, ts_str, code, chars)
	output = get_cat_qa_gpt_4(prompt_str)
	res = output[0]
	prompt_res = output[1]

	# If output length is small, then ignore.
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
	dataset_loc = sys.argv[1]
	dataset = data_loader_mike(dataset_loc)
	to_start = int(sys.argv[2])
	to_end = int(sys.argv[3])

	if to_end == -1:
		to_end = len(dataset)

	for i in range(to_start, to_end):
		try:
			dval = dataset[i]
			get_cat_ques(dval, i)
			print("Reached: "+str(i))
		except:
			print("An exception occurred: "+str(i))

if __name__=="__main__": 
	main() 
