import os
from typing import Dict, Optional, List

import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from random import shuffle

from src.models.tasks.tasks import Task, MultimodalMixin
from src.utils import write_jsonl, read_jsonl, get_logger
from src.models.tasks.utils import ListDataset, TokenizePadAndCollate, simple_list_collate

logger = get_logger(__name__)

# Is this right? Should I be tokenizing elsewhere?
class MultimodalDataset(ListDataset):
    def __init__(self, data, tokenizer = None, 
                 context_columns=["context"],
                 ts_column="ts",
                 label_column="label",
                 context_prefix="",
                 index_from_options=False,
                 **kwargs):
        
        self.tokenizer = tokenizer
        self.index_from_options = index_from_options
        
        self.context_columns = context_columns
        self.ts_column = ts_column
        self.label_column = label_column
        self.context_prefix = context_prefix

        super().__init__(data)

    def __getitem__(self, index) -> Dict:
        data = super().__getitem__(index)
        context = self.context_prefix + " ".join([str(data[col]) for col in self.context_columns])
        if self.index_from_options:
            options = data["options"]
            label = options[data["answer_index"]]
        else:
            label = data[self.label_column]
            
        return {
            "context" : context,
            "ts" : np.array(data[self.ts_column]),
            "label" : label,
        }
    
class MultimodalMCQDataset(ListDataset):
    def __init__(self, data, tokenizer, 
                 context_prefix="",
                 ts_column="ts",
                 label_column="source",
                 options_column="options",
                 shuffle_labels=False,
                 context_columns=["context"],
                 format_abc_mcq=False,
                 **kwargs):
        
        self.tokenizer = tokenizer
        
        self.context_prefix = context_prefix
        self.ts_column = ts_column
        self.label_column = label_column
        self.options_column = options_column
        self.shuffle_labels = shuffle_labels
        self.context_columns = context_columns
        self.format_abc_mcq = format_abc_mcq

        super().__init__(data)

    def __getitem__(self, index) -> Dict:
        data = super().__getitem__(index)
        options = data[self.options_column]

        context = "\n".join([str(data[col]) for col in self.context_columns])
        context = self.context_prefix + context
        
        if self.shuffle_labels:
            shuffle(options)
        
        if "answer_index" in data:
            label_index = data["answer_index"]
            if self.shuffle_labels:
                print("WARNING: You are relying on a precomputed answer index but also shuffling the labels. This is probably not what you want.")

        else:
            label_index = options.index(data[self.label_column])
        
       
        if self.format_abc_mcq:
            full_options = [f"{chr(ord('A') + i)}) {x}" for i, x in enumerate(options)]
            options = [f"{chr(ord('A') + i)}" for i, x in enumerate(options)]
            context = context + "\n" + "\n".join(full_options)


        return {
            "context" : context,
            "ts" : data[self.ts_column],
            "label" : chr(ord('A') + label_index),
            "options" : options,
            "label_index" : label_index
        }

class MultimodalTask(Task, MultimodalMixin):

    def __init__(self, ts_column:str = "series",
                       context_columns:List[str] = ["description_tiny", "metadata"],
                       label_column:str = "description",
                       cache_path:str = "data/processed/synthetic_descriptions",
                       batch_size:int = 16,
                       context_prefix:str = "",
                       index_from_options:bool = False,
                       **kwargs):
        
        self.cache_path = cache_path
        self.ts_column = ts_column
        self.context_columns = context_columns
        self.label_column = label_column
        self.context_prefix = context_prefix
        self.index_from_options = index_from_options

        self.tokenizer = None #TODO Add if ever needed
        self.batch_size = batch_size
        
        super(Task, self).__init__()
        super(MultimodalMixin, self).__init__(**kwargs)
     
    def get_dataloader(self, partition):
        data = self.load(partition)
        dataset = MultimodalDataset(data, tokenizer = self.tokenizer,
                                    context_columns=self.context_columns,
                                    ts_column=self.ts_column,
                                    label_column=self.label_column,
                                    context_prefix=self.context_prefix,
                                    index_from_options=self.index_from_options)
        
        if self.tokenizer:
            data_collator = TokenizePadAndCollate(tokenizer=self.tokenizer,
                                                features_to_tokenize=["context", "label", "options"])
        else:
            data_collator = simple_list_collate
            
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator,
        )
    
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val")
    
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test")
    
    def load(self, partition):
        path = os.path.join(self.cache_path, partition + ".json")
        if not os.path.exists(path):
            logger.info("Attempting to cache data to {}".format(self.cache_path))
            if hasattr(self, "cache"):
                self.cache()
            else:
                raise NotImplementedError("You need to implement a cache method if the data doesn't already exist")
        return read_jsonl(path)
    
class MultimodalMCQTask(MultimodalTask):

    def __init__(self, num_classes:int,
                 options_column:str = "options",
                 format_abc_mcq:bool = False,
                 shuffle_labels: bool = False,
                 **kwargs):
        
        self.num_classes = num_classes
        self.options_column = options_column
        self.format_abc_mcq = format_abc_mcq
        self.shuffle_labels = shuffle_labels

        super().__init__(**kwargs)
    
    def get_dataloader(self, partition):
        data = self.load(partition)
        dataset = MultimodalMCQDataset(data, tokenizer = self.tokenizer,
                                        context_columns=self.context_columns,
                                        context_prefix=self.context_prefix,
                                        ts_column=self.ts_column,
                                        format_abc_mcq=self.format_abc_mcq,
                                        label_column=self.label_column,
                                        options_column=self.options_column,
                                        shuffle_labels=self.shuffle_labels)
        
        if self.tokenizer:
            data_collator = TokenizePadAndCollate(tokenizer=self.tokenizer,
                                                features_to_tokenize=["context", "label", "options"])
        else:
            data_collator = simple_list_collate
            
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=data_collator)
    

class SquareOrSine(MultimodalMCQTask):

    def cache(self):
        np.random.seed(0)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)        

        N_SAMPLES_TRAIN=10000
        N_SAMPLES_VAL=1000
        N_SAMPLES_TEST=1000

        partitions = [("train", N_SAMPLES_TRAIN), ("val", N_SAMPLES_VAL), ("test", N_SAMPLES_TEST)]
        
        def _cache_partition(partition, n_samples):
            data = []
            for _i in range(n_samples):

                if np.random.rand() > 0.5:
                    source = "square"
                    desc = "This synthetic time series represents a square wave with additive noise."
                    wave = self._square_wave()

                else:
                    source = "sine"
                    desc = "This synthetic time series represents a sine wave with additive noise."
                    wave = self._sine_wave()

                data.append({
                "context": "",
                "signal": wave,
                "label": source,
                "options": ["square", "sine"],
                })
            
            with open(os.path.join(self.cache_path, partition + ".json"), "w") as f:
                write_jsonl(f, data)
               
        list(map(lambda x: _cache_partition(*x), partitions))

    def _square_wave(self):
        # Generate a square wave with random amplitude and frequency
        # and additive noise
        n = np.random.randint(1, 10)
        amplitude = np.random.uniform(0, 1)
        frequency = np.random.uniform(5, 20)
        noise = np.random.uniform(0, 1,1000)
        x = np.linspace(0, 1, 1000)
        y = amplitude * np.sin(2 * np.pi * frequency * x) 
        y = np.where(y > 0, 1, -1) + noise
        return y

        
    def _sine_wave(self):
        # Generate a square wave with random amplitude and frequency
        # and additive noise
        n = np.random.randint(1, 10)
        amplitude = np.random.uniform(0, 1)
        frequency = np.random.uniform(5, 20)
        noise = np.random.uniform(0, 1,1000)
        x = np.linspace(0, 1, 1000)
        y = amplitude * np.sin(2 * np.pi * frequency * x) 
        y += noise
        return y

