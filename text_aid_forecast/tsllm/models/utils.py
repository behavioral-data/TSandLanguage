from tsllm.models.gpt import GPTmodel
from tqdm import tqdm
from tsllm.serialize import  deserialize_str , ori_scale_deserialize
import numpy as np 
import pandas as pd

'''
    Note that : 
    This script references https://github.com/ngruver/llmtime and https://arxiv.org/pdf/2310.07820.pdf. Thank you for your work.
'''
STEP_MULTIPLIER = 1.2 
    
def load_model_by_name(config):
    if config.model.name in ['gpt-3.5-turbo','gpt-4']:
        return GPTmodel(config=config) 
                    
def get_output_format(preds , test , results_list , model_name , input_strs ):
    
    samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
    medians = [sample.median(axis=0) for sample in samples]
    samples = samples if len(samples) > 1 else samples[0]
    medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model_name,
        },
        'completions_list': results_list,
        'input_strs': input_strs,
    }
    return out_dict

def get_predict_results(model , input_strs  , test, describtion,  
                        config, batch_size, num_samples, scalers=None  ):
    results_list = []
    batch_preds = []
    for input_str in tqdm(input_strs):
        res = model.run(input_str , describtion , config.model.test_len*STEP_MULTIPLIER , batch_size ,num_samples , config.model.temp ) 
        results_list.append(res)
    for completions, scaler in zip(results_list, scalers):
        preds = []
        for completion in completions:
            # Convert the output string to a numpy array. 
            if scaler is not None :
                deserialized_pred = deserialize_str(completion, config.model.settings, ignore_last=False, steps=config.model.test_len)
            else :
                deserialized_pred = ori_scale_deserialize(completion)

            #  Ensure the forecasting output length matches the ground-truth length.
            pred = handle_prediction(deserialized_pred , expected_length=config.model.test_len, strict=False)
            
            # If there is a rescaling operation, restore the scale.
            if (pred is not None) and (scaler is not None)  : 
                preds.append(scaler.inv_transform(pred))
            else :
                preds.append(pred)
                
        # The batch_size here is 1, preds contian 20 predicted results 
        batch_preds.append(preds)
        
    # Package the results
    out_dict = get_output_format(batch_preds, test , results_list , config.model.name , input_strs )
    return out_dict 

def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
                return None
            else:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]