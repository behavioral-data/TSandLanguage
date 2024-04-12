import hydra
from omegaconf import DictConfig
from tsllm.models.utils import load_model_by_name , get_predict_results 
from tsllm.datasets import get_datasets 
from tsllm.pre_processing import pre_processing 
from tsllm.utils import build_save_path  , is_completion
import os , pickle , time
    
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
    
    datasets = get_datasets(config)
    print('Number of datasets:' , len(datasets))
    
    model = load_model_by_name(config)
    
    num_samples = 20 if 'gpt' in config.model.name else 96 
    batch_size =  0  if 'gpt' in config.model.name else 6
        
    scalers = None 
    save_dir = build_save_path(config)
    
    for dsname,data in datasets.items():
        if is_completion(save_dir , dsname ) : continue
        outs_dict = {}
        train, test , description= data
        print(train , ' -- tes tLen:' , len(test) )
        _, input_strs ,  scalers , test  = pre_processing(train, test , description , config , model.tokenizer )
        print(input_strs)
        try:
            out = get_predict_results(model , input_strs  , test , description  , config, batch_size, num_samples, scalers = scalers )
            outs_dict[config.model.name] = out
        except Exception as e:
            print(f"Failed {dsname} {config.model.name}" + str(e) )
            continue 
        with open(f'{save_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(outs_dict,f)

if __name__ == "__main__":
    run()
