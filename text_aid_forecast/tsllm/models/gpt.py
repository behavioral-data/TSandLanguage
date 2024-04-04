import tiktoken
import numpy as np
import torch 
import openai
import os 
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ["OPENAI_API_BASE"]

openai.api_version = "2023-09-01-preview"
openai.api_type = "azure"
DEPLOYMENT = "gpt-4"
    
class GPTmodel(torch.nn.Module):
    '''
        Note that : 
        This script references https://github.com/ngruver/llmtime and https://arxiv.org/pdf/2310.07820.pdf. Thank you for your work.
    '''
    def __init__(self, config):
        super(GPTmodel, self).__init__()
        self.task = config.experiment.task 
        self.settings  = config.model.settings
        self.model_name = config.model.name
        self.tokenizer = None 
        
    def run(self , input_str , description , steps, batch_size, num_samples , temp ):
        
        if self.task == 'forecast': 
            return self.forecast(input_str  , description  , steps  , num_samples, temp)
        
    def forecast(self, input_str  , description  , steps  , num_samples, temp):
        """
        num_samples: Generate num_samples different time series to help get an average.
        steps : prediction length 
        Generate text completions from GPT using OpenAI's API.
        """

        #  To prevent GPT-3 from producing unwanted tokens
        #  What is allowed : ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '-']
        avg_tokens_per_step = len(self.tokenize_fn(input_str, self.model_name)) / len(input_str.split(self.settings['time_sep']))
        logit_bias = self.get_logit_bias()
        chatgpt_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        if description != '' : 
            description = "This is the description of the time series you are predicting. Understanding it will help with your prediction: " + description 
        extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            # model='gpt4-1106',
            deployment_id=DEPLOYMENT,
            messages=[
                    {"role": "system", "content": chatgpt_sys_message},
                    {"role": "user", "content": description + extra_input+input_str+self.settings['time_sep']}
                ],
            max_tokens=int(avg_tokens_per_step*steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples, # Generate num_samples different time series to help get an average.
        )
        return [choice.message.content for choice in response.choices]
        
    def get_logit_bias(self):
        # define logit bias to prevent GPT from producing unwanted tokens
        logit_bias = {}
        allowed_tokens = [self.settings['bit_sep'] + str(i) for i in range(self.settings['base'])] 
        allowed_tokens += [self.settings['time_sep'], self.settings['plus_sign'], self.settings['minus_sign']]
        allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
        if (self.model_name not in ['gpt-3.5-turbo','gpt-4']): # logit bias not supported for chat models
            logit_bias = {id: 30 for id in self.get_allowed_ids(allowed_tokens, self.model_name)}
        return logit_bias
    
            
    def tokenize_fn(self, str, model):
        """
        This function is to help get the length of input 

        Args:
            str (list of str): str to be tokenized.
            model (str): Name of the LLM model.

        Returns:
            list of int: List of corresponding token IDs.
        """
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode(str)

    def get_allowed_ids(self, strs, model):
        """
        This function is help to limit the output tokens of GPT, to prevent it from
        generating data out of time series. 
        
        Args:
            strs (list of str): strs to be converted.
            model (str): Name of the LLM model.

        Returns:
            list of int: List of corresponding token IDs.
        """
        encoding = tiktoken.encoding_for_model(model)
        ids = []
        for s in strs:
            id = encoding.encode(s)
            ids.extend(id)
        return ids
        
    