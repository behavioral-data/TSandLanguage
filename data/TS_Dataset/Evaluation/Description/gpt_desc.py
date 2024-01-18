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
import base64
from pathlib import Path

# GPT Functions ---------------->
openai.api_key = "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"

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def send_to_gpt3(sendpromt):
	completion = openai.ChatCompletion.create(
		model="gpt-3.5-turbo-16k",
		max_tokens=1000,
		messages=[{"role": "user", "content": sendpromt}])
	return completion

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def send_to_gpt4(sendpromt):
	completion = openai.ChatCompletion.create(
		model="gpt-4-1106-preview",
		max_tokens=1000,
		temperature=0.4,
		messages=[{"role": "user", "content": sendpromt}])
	return completion

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def send_to_gptv(sendpromt, base64_image):
	# OpenAI API Key
	openai.api_key = "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"
	content = [{"type": "text","text": sendpromt},{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
	completion = openai.ChatCompletion.create(
		model="gpt-4-vision-preview",
		max_tokens=3000,
		temperature=0.4,
		messages=[{"role": "user", "content": content}])
	return completion

def verbalizer(gpt_output):
	message = gpt_output['choices'][0]['message']['content']
	return message

# End GPT Functions ---------------->

def eval_gptv(data_prompt, question_prompt, model, base64_image):
	init_text = "You have received a time series. Your task is to select the best description or what you believe is the most accurate depiction of what is happening in the time series. Provide the output in the format: `{Answer: 'correct option'}`. Specifically, if the accurate option is option B, return the output as `{Answer: 'B'}`. Only include the alphabet corresponding to the correct option; do not provide the option text. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Avoid adding any extra text."
	gpt_txt = init_text+'\n\n'+data_prompt+'\n'+question_prompt
	gpt_res = verbalizer(send_to_gptv(gpt_txt, base64_image))
	return gpt_res


def eval_gpt4(data_prompt, question_prompt, model):
	init_text = "You have received a time series. Your task is to select the best description or what you believe is the most accurate depiction of what is happening in the time series. Provide the output in the format: `{Answer: 'correct option'}`. Specifically, if the accurate option is option B, return the output as `{Answer: 'B'}`. Only include the alphabet corresponding to the correct option; do not provide the option text. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Avoid adding any extra text."
	gpt_txt = init_text+'\n\n'+data_prompt+'\n'+question_prompt
	gpt_res = verbalizer(send_to_gpt4(gpt_txt))
	return gpt_res


def eval_gpt3(data_prompt, question_prompt, model):
	init_text = "You have received a time series. Your task is to select the best description or what you believe is the most accurate depiction of what is happening in the time series. Provide the output in the format: `{Answer: 'correct option'}`. Specifically, if the accurate option is option B, return the output as `{Answer: 'B'}`. Only include the alphabet corresponding to the correct option; do not provide the option text. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Avoid adding any extra text."
	gpt_txt = init_text+'\n\n'+data_prompt+'\n'+question_prompt
	gpt_res = verbalizer(send_to_gpt4(gpt_txt))
	return gpt_res


def opt_2_char(opt):
	opt_chars = ['A', 'B', 'C', 'D']
	return opt_chars[opt]


def parse_output(gpt_res, model):
	delimits = string.punctuation
	preds = []
	exps = []
	gpt_res = gpt_res.replace('\r', '').replace('\n', '').replace('\\', '')
	gpt_res = re.findall(r'\{.*?\}', gpt_res)[0].split(",")
	preds = gpt_res[0].split(":")[1].translate(str.maketrans('', '', delimits)).strip()
	
	return preds


def ques_2_str(options):
	# Format: descriptions: first is always right.
	cor = options[0]
	np.random.shuffle(options)
	cor_ind = options.index(cor)
	prompt = ''

	for i in range(len(options)):
		opt = options[i].replace('\r', '').replace('\n', '').replace('\\', '').strip()
		prompt += 'Option '+opt_2_char(i)+': '+opt+'\n'

	return prompt, options, cor_ind


def prep_prompt_only_ts(ts):
	prompt_str = 'Time Series: '+ts
	return prompt_str


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


def load_image(uuid):
	image_path = '../../ts_imgs/'+str(uuid)+'.png'
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def prep_evaluation_desc(dval, model='gpt4'):
	ts_str = ts2str(dval['series'])
	data_prompt = prep_prompt_only_ts(ts_str)

	question_prompt, options, cor_ind = ques_2_str(dval['options'])
	cor_ind = opt_2_char(cor_ind)
	
	if model == 'gpt4':
		output = eval_gpt4(data_prompt, question_prompt, model)
	elif model == 'gptv':
		base64_image = load_image(dval['uuid'])
		output = eval_gptv(data_prompt, question_prompt, model, base64_image)
	else:
		output = eval_gpt3(data_prompt, question_prompt, model)

	preds = parse_output(output, model)

	if len(preds) != len(cor_ind):
		return []
	return [preds, cor_ind]


def index_dict_uuid(dataset):
	uuid_index = {}
	for i in range(len(dataset)):
		uuid = dataset[i]['uuid']
		if uuid not in uuid_index:
			uuid_index[uuid] = []
		uuid_index[uuid].append(i)
	return uuid_index


def load_data(file):
	dataset = []
	with open(file) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset


def get_file_with_ind(ts_list, ind):
	for ts in ts_list:
		if ts.split('_')[0] == str(ind):
			return ts

def absent_or_blank(save_file):
	my_file = Path(save_file)
	if my_file.is_file():
		return os.stat(save_file).st_size == 0
	return True

def get_results():
	# Data Loading
	# Length = 225
	folder_loc = "TS_Questions/"
	to_start = int(sys.argv[1])
	to_end = int(sys.argv[2])
	model = sys.argv[3]
	ts_list = os.listdir(folder_loc)

	if to_end == -1:
		to_end = len(ts_list)

	for i in range(to_start, to_end):
		file = get_file_with_ind(ts_list, i)
		file_name = 'Results/'+str(file)+'_'+model+'.p'
		if absent_or_blank(file_name) == False:
			continue

		# try:
		dval = load_data(folder_loc+file)[0]
		results = prep_evaluation_desc(dval, model)
		cat_list = 'summary'
		
		if results == []:
			print("Length error")
			continue
		
		preds, cor_ind = results[0], results[1]
		pickle.dump([dval['description'], cat_list, preds, cor_ind], open(file_name, 'wb'))
		# except:
		# 	print("Error")

if __name__=="__main__": 
	get_results() 
