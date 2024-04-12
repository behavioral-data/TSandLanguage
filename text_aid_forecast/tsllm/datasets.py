import pandas as pd
import os 

def get_datasets(config):
    if "forecast" in config.experiment.task :
        get_datasets(config,testfrac=0.2 )
        
def get_datasets(config,testfrac=0.2 ):
    # v2.jsonl is the dataset you download. It may have other names. 
    # Please put your dataset under target Path : config.experiment.data_path
    data_list = pd.read_json(os.path.join(config.experiment.data_path ,'v2.jsonl' ), lines=True)
    datas = []
    data_indexs =[]
    for _, row in data_list.iterrows():    
        try:  
            series = pd.Series(row['series'])
            splitpoint = int(len(series)*(1-testfrac))
            train = series.iloc[:splitpoint]
            test  = series.iloc[splitpoint:]
            ts_info =''
            if config.experiment.description_type !='':
                if 'description' in config.experiment.description_type : 
                    # Prepend Description
                    ts_info = row['description']+ ' '
                if 'characteristics' in config.experiment.description_type : 
                    # Prepend Characteristics
                    ts_info += row['characteristics']+ ' '
                if 'metadata' in config.experiment.description_type :
                    # Prepend Metadata
                    meta_info = row['metadata']
                    ts_info +=  "The time series was collected between {} and {} with a collection frequency of {}, and the data Unit is \"{}\".  You will predict the next {} data points.".format(meta_info['start'] , meta_info['end'] , meta_info['frequency']  , meta_info['units'] , len(test)  )
            datas.append((train, test , ts_info ))
            data_indexs.append(row['uuid'])
            if len(data_indexs) >= (config.experiment.num_of_sampels) :  break # no need loading the whole set
        except: 
            continue
    datasets = dict(zip(data_indexs,datas))
    return datasets