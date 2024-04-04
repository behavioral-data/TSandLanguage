# print(os.environ['OPENAI_API_KEY'] , os.environ["OPENAI_API_BASE"] )
# exit()
'''
    export OPENAI_API_KEY=2bbc61749c3647e29091f13a81d6767e
    export OPENAI_API_BASE=https://ts-language-oai.openai.azure.com/
    # openai.api_base = "https://ts-language-oai.openai.azure.com/"
    # openai.api_key = '2bbc61749c3647e29091f13a81d6767e'
'''
        
'''
    python3 ./tsllm/main.py experiment=llmtime_wo_context model=gpt-4   *   
    python3 ./tsllm/main.py experiment=llm_wo_context model=gpt-4   
        
    python3 ./tsllm/main.py experiment=llmtime_wi_Ca model=gpt-4     
    python3 ./tsllm/main.py experiment=llmtime_wi_Ch model=gpt-4     
    python3 ./tsllm/main.py experiment=llmtime_wi_ChMe model=gpt-4     
    python3 ./tsllm/main.py experiment=llmtime_wi_CaMe model=gpt-4     
    python3 ./tsllm/main.py experiment=llmtime_wi_all model=gpt-4    * 
    python3 ./tsllm/main.py experiment=llm_wi_all model=gpt-4     
    
    export PYTHONPATH="${PYTHONPATH}:/Users/mingtiantan/Desktop/textAid_LLM_Forecast/"
'''

