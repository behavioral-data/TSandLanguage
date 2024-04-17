import pdb
import os
import sys
import numpy as np
import pickle
import time
import openai
import backoff
import json
import string
import re
from pathlib import Path
import random
from random import choices
import matplotlib.pyplot as plt

'''
In this file, we query GPT to perform the following: 
	1. Create multiple different counterfactual scenarios for every time series. 
	2. For each counterfactual scenario, create code that generates the updated time series.
'''
def send_to_gpt(model, gpt_sys, gpt_usr):
	openai.api_base = "https://ts-language-oai.openai.azure.com/"
	openai.api_key = dotenv_values(".env")["OPENAI_API_KEY_AZURE"]
	openai.api_version = "2023-09-01-preview"
	openai.api_type = "azure"
	DEPLOYMENT = "gpt-4"

	MAX_RETRIES = 10
	current_tries = 1
	while current_tries <= MAX_RETRIES:
		try:
			response = openai.ChatCompletion.create(engine=model, messages=[{"role": "system", "content": gpt_sys}, {"role": "user", "content": gpt_usr}],max_tokens=1000)
			break
		except Exception as e:
			print('openai retrying, error:', str(e))
			time.sleep(10)
			current_tries += 1

	response_text = response['choices'][0]['message']['content']
	return response_text

def plotter(original, question, new_series, app):
	x = np.arange(len(original))
	plt.plot(x, original, "-b", label="O")
	plt.title(question)
	plt.plot(x, new_series, "-r", label="C")
	plt.legend(loc="upper left")
	plt.savefig("Plots/"+str(app)+'.png')
	plt.close()

def get_best_series(old_series, new_series, maintain_len = 0):
	if maintain_len == 0:
		return new_series, 0
	else:
		if len(new_series) == len(old_series):
			return new_series, 0

		elif len(new_series) > len(old_series):
			indices = random.sample(range(len(new_series)), len(old_series))
			res = 1
		else:
			indices = choices(len(new_series), k=len(old_series))
			res = -1

		indices = sorted(indices)
		new_series = [new_series[i] for i in indices]
		return new_series, res

def parse_code(gpt_res):
	new_code = gpt_res.split('```')[1].split('python')[1]
	return new_code

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

def clean_str(sample):
	delimits = string.punctuation.replace('?','').replace('.','')
	return sample.translate(str.maketrans('', '', delimits)).strip()

def prep_mcq_cat(ques_dump):
	ques_dump = ques_dump.replace('\r', '').replace('\n', '').replace('\\', '')
	ques_dump = re.findall(r'\{.*?\}', ques_dump)
	ques_list = []
	for cur_line in ques_dump:
		try:
			cat = clean_str(cur_line.split('category')[1].split('question')[0])
			ques = clean_str(cur_line.split('question')[1].split('answer')[0])
			ans = clean_str(cur_line.split('answer')[1].split('incorrect')[0])
			opt1 = clean_str(cur_line.split('incorrect answer 1')[1].split('incorrect')[0])
			opt2 = clean_str(cur_line.split('incorrect answer 2')[1].split('incorrect')[0])
			opt3 = clean_str(cur_line.split('incorrect answer 3')[1].split('}')[0])
			options = [ans, opt1, opt2, opt3]
			cor = options[0]
			np.random.shuffle(options)
			cor_ind = options.index(cor)
			ques_list.append([cat, ques, options, cor_ind])
		except:
			continue
	return ques_list

def generated_series_okay(series):
	if isinstance(series, list):
		series = np.array(series)
	if series.ndim != 1:
		return False
	if len(series) > 15000:
		return False
	if not np.issubdtype(series.dtype, np.number):
		return False
	if np.isnan(series).any():
		return False
	if np.isinf(series).any():
		return False
	if np.iscomplex(series).any():
		return False
	if np.max(series) > 1e9:
		return False
	return True

def parse_questions(ques_dump):
	ques_dump = re.findall(r'\{.*?\}', ques_dump)
	ques_list = []
	for cur_line in ques_dump:
		ques_list.append(clean_str(cur_line))
	return ques_list

def cs_gen(dval):
	sys_pr = all_prompts('generate_ques')
	user_pr = 'Description: '+dval['description_short']+'\nCode:'+dval['generator']
	ret_list = parse_questions(send_to_gpt('gpt-4', sys_pr, user_pr))
	return ret_list

def absent_or_blank(save_file):
	my_file = Path(save_file)
	if my_file.is_file():
		return os.stat(save_file).st_size == 0
	return True

def create_code_dumps(dval, ques, fname):
	sys_pr = all_prompts('generate_code')
	user_pr = "Description: "+dval['description_short']+"\nOriginal Code: "+dval['generator']+"\n\nChange: "+ques+"\n"
	while True:
		try:
			code_res = send_to_gpt('gpt-4', sys_pr, user_pr)
			new_code = parse_code(code_res)
			exec(new_code, globals())
			new_series = generate_series()
			if generated_series_okay(new_series) != True:
				continue
			else:
				break
		except:
			continue

	new_series, res = get_best_series(dval['series'], new_series)
	print('Done for: '+fname)
	pickle.dump([new_series, code_res, res, ques], open('Dumps/'+fname+'_Code.p', 'wb'))
	return new_series, res, code_res

def main():
	folder_loc = "Splits/"
	to_start = int(sys.argv[1])
	to_end = int(sys.argv[2])
	ts_list = os.listdir(folder_loc)

	if to_end == -1:
		to_end = len(ts_list)

	for i in range(to_start, to_end):
		dval = data_loader_mike(folder_loc+str(i)+'.json')[0]
		uuid = dval['uuid']
		names = [uuid+'_'+str(q)+'_Code.p' for q in range(5)]
		flag = 0
		for n in names:
			if absent_or_blank('Dumps/'+n) == False:
				flag = 1
				break
		if flag == 1:
			continue
		try:
			questions = cs_gen(dval)
		except:
			print('Questions Error: '+uuid)
			continue
		for q in range(len(questions)):
			name = uuid+'_'+str(q)+'_Code.p'
			new_series, res, code_res = create_code_dumps(dval, questions[q], name)
		

def all_prompts(pr_type):
	if pr_type == 'generate_ques':
		sys_pr = "You have been given a description of a time series and a code that generates the time-series. Your task is to create five counterfactual questions that someone can ask regarding this time series. Try to formulate questions that are distinct from each other. Additionally, ensure that the questions aim to bring about significant changes to the time series. Make sure that the new time series can be easily generated by modifying the code and do not ask extremely difficult questions. Format the output as follows:'{'question'}'\n, with each new question starting on a new line.  The counterfactual questions should explore hypothetical scenarios and involve 'What-if' type inquiries. The questions should not include values directly from the original time series or code. For instance, 'What if the start was 25 units' is preferred over 'What if the start was 25 units instead of 20 units?'. Avoid referencing random noise, the random number generator, its mean, or variance in any question. Do not generate any additional text. "
	elif pr_type == 'generate_code':
		sys_pr = "Generate a new Python code for a time series based on the given code and description. The user will specify a change in the time series, and you should produce the updated code using the function name 'generate_series'. Always ensure the length of the time series remains unchanged. This is hard constraint that should not be violated. Keep realistic expectations and ensure the length of the time series remains unchanged. For example, (1) keep the rate of change consistent rather than the actual values. (2) Understand what changes the user's suggestion can make to the time series and then update the code accordingly. (3) Given a time series code, you have the freedom, and in some cases, the obligation, to modify any pre-defined maximum or minimum values specified in the original code to accurately represent the desired change. Ensure that the new time series adheres to real-world principles; for instance, maintaining a consistent rate of change under typical conditions. If the change demands that the time series has an offset by some units, then modify this value in the code as well. \nReturn the output in the format ```new code```, where the 'new code' is replaced by the updated code. Try to create code that generates a time-series that is significantly different from the time-series produced by the original code, but with same lengths. Always return the code in a format that can be executed directly using the exec() function. Avoid additional text."
	elif pr_type == 'get_ques_same_len':
		sys_pr = "Given two Python codes for generating time series, the first representing the original time series with a description, and the second presenting a modification of the original time series under specific conditions. Your task is to ask five questions regarding the differences between both time series. Also ask five questions regarding the simmilarity between both time series. Additionally, provide answers to all the questions and three negative or incorrect options. Ask questions regarding the patterns within both time-series, such as how they appear, the rates of change, and any specific differences in trends.  Format the output as follows:'{'category':'difference/simmilarity', 'question':'', 'answer':'', 'incorrect answer 1':'', 'incorrect answer 2':'', 'incorrect answer 3':''}'. Make sure you follow the following rules: 1) Try to ask a few but not all questions regarding the lengths or the number of data-points within both the time-series. (2) Ensure that the questions and answers give the impression of being created independently, in the absence of the code, solely by examining the time series. (3) Do not mention the random noise or random number generator in both the answers and questions. (4) Try to keep the answers short and not very detailed. (5) Ensure that each question and its corresponding answer are presented on the same line, with each new question starting on a new line for a clear and organized format. (6) Ensure that numerical values are present in the answers and the questions wherever possible. Represent all numbers as if they are approximately calculated, using expressions such as 'seems to be' or 'around value'. Avoid unnecessary text and focus on precision."
	elif pr_type == 'get_ques_diff_len':
		sys_pr = "Given two Python codes for generating time series, the first representing the original time series with a description, and the second presenting a modification of the original time series under specific conditions. Your task is to ask five questions regarding the differences between both time series. Also ask five questions regarding the simmilarity between both time series. Additionally, provide answers to all the questions and three negative or incorrect options. Ask questions regarding the patterns within both time-series, such as how they appear, the rates of change, and any specific differences in trends. Format the output as follows:'{'category':'difference/simmilarity', 'question':'', 'answer':'', 'incorrect answer 1':'', 'incorrect answer 2':'', 'incorrect answer 3':''}'. Make sure you follow the following rules: (1) Do not ask question regarding the lengths or the number of data-points within both the time-series. (2) Ensure that the questions and answers give the impression of being created independently, in the absence of the code, solely by examining the time series. (3) Do not mention anything regrarding the random noise or random number generator in both the answers and questions. (4) Try to keep the answers short and not very detailed. (5) Ensure that each question and its corresponding answer are presented on the same line, with each new question starting on a new line for a clear and organized format. (6) Try to add numerical values to answers wherever possible, but make sure you use words such as 'seems to be' or 'around value' so that they appear to be approximate. Avoid unnecessary text and focus on precision."
	return sys_pr

if __name__=="__main__": 
	main() 
