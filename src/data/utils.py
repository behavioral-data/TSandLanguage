import os
import glob 
import json
import pickle
import re

import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.special import softmax
from torch.utils import data
import numpy as np

from src.utils import get_logger

import wandb
from tqdm import tqdm

import torch
from dotenv import dotenv_values


config = dotenv_values(".env")
logger = get_logger(__name__)

DATASET_VERSION="2020-07-15"
main_path = os.getcwd()
RAW_DATA_PATH = os.path.join(main_path,"data","raw","audere","data-export",DATASET_VERSION)
PROCESSED_DATA_PATH = os.path.join(main_path,"data","processed")
DEBUG_DATA_PATH = os.path.join(main_path,"data","debug")

def get_features_path(name):
    if os.environ.get("DEBUG_DATA"): 
        logger.warning("DEBUG_DATA is set, only loading subset of data")
        data_path = DEBUG_DATA_PATH
    else:
        data_path = PROCESSED_DATA_PATH
    return os.path.join(data_path,"features",name+".csv")


def download_wandb_table(run_id,table_name="roc_table",
                 entity="mikeamerrill", project="flu"):
    api = wandb.Api()
    artifact = api.artifact(f'{entity}/{project}/run-{run_id}-{table_name}:latest')
    dir = artifact.download() 
    filenames = list(glob.glob(os.path.join(dir,"**/*.json"),recursive=True))
    data = load_json(filenames[0])
    return pd.DataFrame(data["data"],columns=data["columns"])


def load_results(path):
    results = pd.read_json(path,lines=True)
    logits = pd.DataFrame(results["logits"].tolist(), columns=["pos_logit","neg_logit"])
    softmax_results = softmax(logits,axis=1)["neg_logit"].rename("pos_prob")
    return pd.concat([results["label"],logits,softmax_results],axis=1)

def write_dict_to_json(data,path,safe=True):
    if safe:
        data = {k:v for k,v in data.items() if is_jsonable(v)}

    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def load_json(path):
    with open(path, 'r') as infile:
        return json.load(infile)



def url_from_path(path,filesystem="file://"):
    if path:
        if isinstance(path,str):
            return filesystem + path
        else:
            return [url_from_path(p,filesystem) for p in path]
        

def get_categorial_columns(df):
    """Return a list of columns that are likely to be categorical. Includes columns
    that appear to be boolean, and columns that have less than 10 unique values."""
    categorial_cols = []
    for col in df.columns:
        if df[col].dtype == "bool":
            categorial_cols.append(col)
        elif len(df[col].unique()) < 10:
            categorial_cols.append(col)
    return categorial_cols

def get_numerical_columns(df):
    """Return a list of columns that are likely to be numerical. Includes columns
    that appear to be numeric, and columns that have more than 10 unique values."""
    numerical_cols = []
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            numerical_cols.append(col)
        elif len(df[col].unique()) > 10:
            numerical_cols.append(col)
    return numerical_cols

def filter_cols_regex(regex_list, df):
    cols = []
    for col in df.columns:
        matches = [re.match(regex, col) for regex in regex_list]
        if not any(matches):
            cols.append(col)
    return cols

def df_to_dict_of_rows(df):
    """Convert a dataframe to a dictionary of rows, where the key is the index of the row"""
    return {i: row.values for i, row in df.iterrows()}