import os , sys 
sys.path.append(os.path.abspath('../../'))
from src.data.token_utils import get_scaler , truncate_input
from src.data.serialize import serialize_arr , ori_scale_serialize 
from omegaconf import open_dict
import pandas as pd
import numpy as np 
    
def pre_processing(train, test, describtion  , config , tokenizer  ):
    if 'rescale' in config.experiment.preprocess : 
        return rescale_pre_processing(train, test,  describtion , config , tokenizer )
    if 'ori_scale' in  config.experiment.preprocess :
        return ori_scale_pre_processing(train, test , config  )

def ori_scale_pre_processing(train, test , config ):
    if not isinstance(train, list):
        train = [train]
        test = [test]
        
    # Add a temporary key value to represent the length of the prediction target.
    with open_dict(config):
        config.model.test_len = len(test[0])
        
    assert all(len(t)==config.model.test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'
    input_arrs = [train[i].values for i in range(len(train))]
    input_strs = ori_scale_serialize(input_arrs) 
    return None, input_strs , [None]*len(input_strs) , test
    
def rescale_pre_processing(train, test, describtions , config , tokenizer ):
    '''
        Note that : 
        This script references https://github.com/ngruver/llmtime and https://arxiv.org/pdf/2310.07820.pdf. Thank you for your work.
        
        tokenizer is to help input fit the maximum context length (model`s)
    '''
    if not isinstance(train, list):
        train = [train]
        test = [test]
        describtions = [describtions]
    
    with open_dict(config):
        config.model.test_len = len(test[0])
        
    assert all(len(t)==config.model.test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'
    
    scalers = [get_scaler(train[i].values, alpha=config.model.alpha, beta=config.model.beta, basic=config.model.basic) for i in range(len(train))]

    input_arrs = [train[i].values for i in range(len(train))]
    '''
        Normailize time series, to make rescaled result locate in certain range 
        
        Normalize example : 
            112 -> 0.25917881 
            118 -> 0.27170962 
            .... 

        q= 478.82 ; min_ = -12.099  (q is not the max_value)
        transform     : (x - min_) / q
        inv_transform : x * q + min_ 
    '''
    
    transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
    '''
        Shift the decimal point to ensure that values after rescaling fall within the 0-2000 range as much as possible.
         
        example : 
            0.25917881  -> [0 0 0 ...0 2 5 9] -> 259
            1.05070799  -> [0 0 0 ...1 0 5 0] -> 1050
        input_strs: ['627, 661, 739, 723,....']
    '''
    input_strs = [serialize_arr(scaled_input_arr, config.model.settings) for scaled_input_arr in transformed_input_arrs]
    truncated_input_arr, truncated_input_str = zip(*[truncate_input(input_array, input_str, describtion, config , tokenizer ) for input_array, input_str ,describtion in zip(input_arrs, input_strs , describtions )])

    return truncated_input_arr, truncated_input_str , scalers , test