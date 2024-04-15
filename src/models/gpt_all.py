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
from data_utils import get_scaler , truncate_input
from serialize import serialize_arr, deserialize_str  
import pandas as pd

def send_to_gpt(model, gpt_sys, gpt_usr):
	if model== 'gpt4':
		model = 'gpt4-1106'
	elif model == 'gpt3':
		model = 'gpt-35'
	openai.api_type = "azure"
	openai.api_base = "https://ts-language-oai.openai.azure.com/"
	openai.api_version = "2023-07-01-preview"
	openai.api_key = dotenv_values(".env")["OPENAI_API_KEY_AZURE"]

	MAX_RETRIES = 10
	current_tries = 1
	while current_tries <= MAX_RETRIES:
		try:
			response = openai.ChatCompletion.create(engine=model, messages=[{"role": "system", "content": gpt_sys}, {"role": "user", "content": gpt_usr}],max_tokens=100)
			break
		except Exception as e:
			print('openai retrying, error:', str(e))
			time.sleep(10)
			current_tries += 1

	response_text = response['choices'][0]['message']['content']
	return response_text

# End GPT Functions ---------------->
def eval_gptv(data_prompt, question_prompt, model, base64_image):
	init_text = "You have received a time series and a concise description. Your task is to respond to the associated multiple-choice questions. Provide the output in the format: `{'question number': 'correct option'}`. Specifically, if the accurate option for question #1 is option B, return the output as `{'1': 'B'}`. Make sure that you answer all questions. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Present each question and the selected answer in the one line question with the result of every MCQ in a new line. Only include the alphabet corresponding to the correct option; do not provide the option text. Avoid adding any extra text."
	gpt_txt = init_text+'\n\n'+data_prompt+'\n'+question_prompt
	gpt_res = verbalizer(send_to_gptv(gpt_txt, base64_image))
	return gpt_res

def eval_gpt4(data_prompt, question_prompt, model):
	init_text = "You have received a time series and a concise description. You have alo been given a new time-series that is a counterfactual version of the former. The new version answers the question"++"Your task is to respond to the associated multiple-choice questions. Provide the output in the format: `{'question number': 'correct option'}`. Specifically, if the accurate option for question #1 is option B, return the output as `{'1': 'B'}`. Make sure that you answer all questions. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Present each question and the selected answer in the one line question with the result of every MCQ in a new line. Only include the alphabet corresponding to the correct option; do not provide the option text. Avoid adding any extra text."
	gpt_txt = init_text+'\n\n'+data_prompt+'\n'+question_prompt
	gpt_res = verbalizer(send_to_gpt4(gpt_txt))
	return gpt_res

def eval_gpt3(data_prompt, question_prompt, model):
	# init_text = 
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
	# if model == 'gpt4' or model == 'gptv':
	# 	gpt_res = gpt_res.split('\n')
	# 	for i in range(len(gpt_res)):
	# 		if len(gpt_res[i]) < 5:
	# 			continue
	# 		preds.append(gpt_res[i].split(":")[1].translate(str.maketrans('', '', delimits)).strip())
	# else:
	gpt_res = gpt_res.replace('\r', '').replace('\n', '').replace('\\', '')
	gpt_res = re.findall(r'\{.*?\}', gpt_res)[0].split(",")
	for i in range(len(gpt_res)):
		if len(gpt_res[i]) < 3:
			continue
		preds.append(gpt_res[i].split(":")[1].translate(str.maketrans('', '', delimits)).strip())
	return preds


def ques_2_str(ques_dump):
	# Format: ques, options, cor_ind
	cor_inds = []
	ques_list = []
	ret_str = ''
	for i in range(len(ques_dump)):
		ind_str = ''
		ind_str += 'Question #'+str(i+1)+': '+ques_dump[i][0]+'\n'
		for j in range(len(ques_dump[i][1])):
			ind_str += 'Option '+opt_2_char(j)+': '+ques_dump[i][1][j]+'\n'
		ret_str = ret_str+'\n\n'+ind_str
		cor_inds.append(ques_dump[i][2])
	return ret_str.strip(), cor_inds


def prep_prompt(desc, met, ts=''):
	if ts != '':
		prompt_str = 'Description: '+desc+'\n'+'MetaData: '+met+'\n'+'Time Series: '+ts
	else:
		prompt_str = 'Description: '+desc+'\n'+'MetaData: '+met+'\n'
	return prompt_str


def rescale_processing(target_series):
    '''
	    target_series shuould be a numpy array with shape (len,) or a list
    '''
    train = [pd.Series(target_series)]
    # Create a unique scaler for each time series sample 
    alpha , beta , basic = 0.3, 0.3, True 
    settings = { 'base': 10 ,
    'prec': 3 ,
            'signed': True ,
            'fixed_length': False ,
            'max_val': 10000000.0 ,
            'time_sep': ', '  ,
            'bit_sep': '' ,
            'plus_sign': '' ,
            'minus_sign': '-' ,
            'half_bin_correction': True ,
            'decimal_point': '' ,
            'missing_str': ' Nan'
    }
    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]
    '''
        Normalize example : 
            112 -> 0.25917881 
            118 -> 0.27170962 
            .... 

        q= 478.82 ; min_ = -12.099
        transform     : (x - min_) / q
        inv_transform : x * q + min_ 
    '''
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
    '''
        This idea is from https://arxiv.org/pdf/2310.07820.pdf (NYU)
        example : 
            0.25917881  -> [0 0 0 ...0 2 5 9] -> 259
            1.05070799  -> [0 0 0 ...1 0 5 0] -> 1050
             
        input_strs: ['627, 661, 739, 723,....']
    '''    
    input_strs = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in transformed_input_arrs]
    return input_strs


def load_image(uuid):
	image_path = '../../ts_imgs/'+str(uuid)+'.png'
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def prep_evaluation(dval, mcq_list, with_ts, model, count_question):
	desc_t = dval['description_tiny'].replace('\n', '').strip()
	met = ''.join('{}: {}, '.format(key, val) for key, val in dval['metadata'].items())
	met = met.replace('\n', '').strip()
	ts_str = rescale_processing(dval['series'])[0]

	question_prompt, cor_inds = ques_2_str(mcq_list)
	cor_inds = [opt_2_char(i) for i in cor_inds]

	if with_ts == 0:
		data_prompt = prep_prompt(desc_t, met)
	else:
		data_prompt = prep_prompt(desc_t, met, ts_str)

	if model == 'gpt4':
		output = send_to_gpt(model, all_prompts('gpt4'), question_prompt)
	elif model == 'gptv':
		base64_image = load_image(dval['uuid'])
		output = send_to_gptv(model, all_prompts('gptv'), question_prompt, base64_image)
	else:
		model = 'gpt3'
		output = send_to_gpt(model, all_prompts('gpt3'), question_prompt)
	
	preds = parse_output(output, model)

	if len(preds) != len(mcq_list):
		return []
	return [preds, cor_inds]

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

def get_dvals_per_uuid(dataset, ts_list):
	# Here the dictionary will save the file-name for new_code and line in dataset
	# Index: filename for questions file
	dval_list = {}
	for ts in ts_list:
		if ts[0] == '.' or ts[0] == '_':
			continue
		uuid = ts.split('_')[0]
		for i in range(len(dataset)):
			if dataset[i]['uuid'] == uuid:
				dval_list[ts] = [dataset[i], ts.replace('Ques', 'Code')]
	return dval_list


def get_results_linear():
	to_start = int(sys.argv[1])
	to_end = int(sys.argv[2])
	model = sys.argv[3]
	wts = int(sys.argv[4])
	dataset = load_data('v2_8787.jsonl')
	if to_end == -1:
		to_end = len(dataset)
	dataset = dataset[to_start:to_end]

	ques_list = os.listdir("Dumps/")
	ques_list = [q for q in ques_list if q.split('_')[-1]=='Ques.p']
	dval_list = get_dvals_per_uuid(dataset, ques_list)

	for k, v in dval_list.items():
		# One more check to ignore .DS_Store
		if k[0] == '.' or k[0] == '_':
			continue
		res_file = k.replace('_Ques.p', '')+'_'+model+'_'+str(wts)+'.p'
		res_file = 'Results/'+res_file
		if absent_or_blank(res_file) == False:
			continue

		dval = v[0]
		ques_dump = pickle.load(open('Dumps/'+k, 'rb'))
		code_dump = pickle.load(open('Dumps/'+v[1], 'rb'))

		mcq_list = []
		cat_list = []
		for i in range(len(ques_dump)):
			mcq_list.append([ques_dump[i][1], ques_dump[i][2], ques_dump[i][3]])
			cat_list.append(ques_dump[i][0])

		results = prep_evaluation(dval, mcq_list, wts, model, code_dump[3])		
		if results == []:
			print("Length error")
			continue
		
		preds, cor_inds = results[0], results[1]
		pickle.dump([mcq_list, cat_list, preds, cor_inds], open(res_file, 'wb'))
		# except:
			# print("Error :"+str(i))

def all_prompts(ptype):
	if ptype == 'gpt3':
		sys_pr = "You have received a time series and a concise description. Your task is to respond to the associated multiple-choice questions. Provide the output in the format: `{'question number': 'correct option'}`. Specifically, if the accurate option for question #1 is option 'B' and for question #2 is option 'A', return the output as `{'1': 'B', '2': 'A'}`. Make sure that you answer all questions. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Present each question and the selected answer for all MCQs in only one line. Only include the alphabet corresponding to the correct option; do not provide the option text. Avoid adding any extra text."
	elif ptype == 'gpt4':
		sys_pr = "You have received a time series and a concise description. You have also been given a new time-series that is a counterfactual version of the former. The new version is a variant of the orignial that answers a given question. Your task is to respond to the associated multiple-choice questions. Provide the output in the format: `{'question number': 'correct option'}`. Specifically, if the accurate option for question #1 is option 'B' and for question #2 is option 'A', return the output as `{'1': 'B', '2': 'A'}`. Make sure that you answer all questions. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Present each question and the selected answer in the one line question with the result of every MCQ in a new line. Only include the alphabet corresponding to the correct option; do not provide the option text. Avoid adding any extra text."
	elif ptype == 'gptv':
		sys_pr = "You have received a time series and a concise description. You have also been given a new time-series that is a counterfactual version of the former. The new version is a variant of the orignial that answers a given question. Your task is to respond to the associated multiple-choice questions. In the plot, the red line indicated the new series, whereas the blue-line indicated the original time-series. Your task is to respond to the associated multiple-choice questions. Provide the output in the format: `{'question number': 'correct option'}`. Specifically, if the accurate option for question #1 is option B, return the output as `{'1': 'B'}`. Make sure that you answer all questions. Make sure that each line begins with a curly bracket '{' and ends with a curly bracket '}'. Present each question and the selected answer in the one line question with the result of every MCQ in a new line. Only include the alphabet corresponding to the correct option; do not provide the option text. Avoid adding any extra text."
	return sys_pr

	return sys_pr
if __name__=="__main__": 
	get_results_linear() 
