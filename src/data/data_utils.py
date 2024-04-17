import torch
import os.path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass

STEP_MULTIPLIER = 1.2

@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.
    
    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.
        
    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:   
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            return (x - min_) / q
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)

def truncate_input(input_arr, input_str , describtion , config, tokenizer=None  ):
    """
    Truncate inputs to the maximum context length for a given model.
    
    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    if tokenizer != None and config.model.context_lengths != None : 
        tokenization_fn = tokenizer
        context_length  = config.model.context_lengths
        input_str_chuncks = input_str.split(config.model.settings['time_sep'] )
        has_truncated = False 
        
        for i in range(len(input_str_chuncks) - 1):
            truncated_input_str = config.model.settings['time_sep'].join(input_str_chuncks[i:])
            
            if not truncated_input_str.endswith(config.model.settings['time_sep']):
                # add separator if not already present
                truncated_input_str += config.model.settings['time_sep']
            
            if describtion != '' : 
                num_descri_tokens =len(tokenization_fn(describtion)) 
            else : 
                num_descri_tokens = 0 
                
            input_tokens = tokenization_fn(truncated_input_str)
            num_series_tokens = len(input_tokens) 
                
            avg_token_length = num_series_tokens / (len(input_str_chuncks) - i)
            num_output_tokens = avg_token_length * config.model.test_len * STEP_MULTIPLIER
            
            num_input_toekns = num_descri_tokens + num_series_tokens
            
            if num_input_toekns + num_output_tokens  <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            has_truncated = True 
            print(f'Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}')
        if config.is_test_context_length:
            return has_truncated
        return truncated_input_arr, truncated_input_str
    else:
        return input_arr, input_str

def hardSigma(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output

def printParams(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    
def makedir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except:
        pass

def build_save_path(config):
    save_dir = f'{config.output_dir}/{config.model.name}-{config.experiment.data_name}-{config.experiment.exp_object}'
    if 'ts_scales' == config.experiment.exp_object : 
        save_dir += '_{}'.format(config.scale_size)
    if not os.path.exists(save_dir) : os.mkdir(save_dir)
    return save_dir
    
def is_completion(save_dir , dsname ):
    if os.path.exists(f'{save_dir}/{dsname}.pkl'):
        print("uuid {} has been finished".format(dsname)) ; 
        return True
    else: 
        return False 

# f98ddf5d-29c0-4e2e-8130-0ceed4d0c2e3
import time 
def is_exception(dsname , exp_object):
    longest_list = [uuid.split('.')[0] for uuid in os.listdir('/p/selfdrivingpj/projects_time/TSLLMs-main/outputs/gpt-4-uw-ts_wi_chara_meta')]
    # longest_list = [uuid.split('.')[0] for uuid in os.listdir('/p/selfdrivingpj/projects_time/TSLLMs-main/outputs/gpt-4-uw-ts_scales_-5')]
    if exp_object =='ts_wi_chara_meta' : return False 
    if dsname not in longest_list : 
        # time.sleep(5)
        if dsname not in longest_list :  
            print('excepiton in ts_wi_chara_meta : ' , dsname)
            return True 
        else : return False
    else : 
        return False 