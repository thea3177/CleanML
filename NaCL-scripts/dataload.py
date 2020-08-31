import pickle
import os
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

def listdir(directory):
    return [d for d in os.listdir(directory) if d[0]!="."]

def get_jobs(data_dir, error_type = 'missing_values', dataset = ['Titanic']):
    jobs = []
    # run all datasets
    if dataset == 'all':  
        datasets = listdir(data_dir)

        for d in datasets:
            error_types = listdir(os.path.join(data_dir, d))
            for e in error_types:
                splits = listdir(os.path.join(data_dir, d, e))
                for s in splits:
                    exps = listdir(os.path.join(data_dir, d, e, s))
                    for ex in exps:
                        jobs.append([data_dir, d, e, s, ex])
    
    # run some specific datasets on some particular error types
    else:
        for d in dataset:
            splits = listdir(os.path.join(data_dir, d, error_type))
            for s in splits:
                exps = listdir(os.path.join(data_dir, d, error_type, s))
                for ex in exps:
                    jobs.append([data_dir, d, error_type, s, ex])
        
    return jobs

def get_jobs_mv(data_dir, error_type = 'missing_values', dataset = ['Titanic']):
    jobs = []
    
    
    for d in dataset:
        splits = listdir(os.path.join(data_dir, d, error_type))
        for s in splits:
            exps = listdir(os.path.join(data_dir, d, error_type, s))

            jobs.append([data_dir, d, error_type, s])
        
    return jobs

def load_data(job):
    with open(os.path.join(*job), "rb") as f:
        data = pickle.load(f)
    return data

def load_result(save_dir, dataset):
    result_path = os.path.join(save_dir, '{}_result.json'.format(dataset))
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
    else:
        result = {}
    return result

def save_result(job, result_dict, model_name, save_dir):
    _, d, e, s, ex = job
    split_seed = s.split("_")[1]
    ex_seed = ex[:-2].split("_")[1]
    key = "{}/v{}/{}/dirty/{}/{}".format(d, split_seed, e, model_name, ex_seed)

    # load old results
    old_result = load_result(save_dir, d)
    old_result[key] = result_dict

    # save result
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_path = os.path.join(save_dir, '{}_result.json'.format(d))
    with open(result_path, 'w') as f:
        json.dump(old_result, f, indent=4)

def save_result_mv(job, seed, result_dict, model_name, save_dir):
    _, d, e, s = job
    split_seed = s.split("_")[1]
    ex_seed = seed
    key = "{}/v{}/{}/dirty/{}/{}".format(d, split_seed, e, model_name, ex_seed)

    # load old results
    old_result = load_result(save_dir, d)
    old_result[key] = result_dict

    # save result
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_path = os.path.join(save_dir, '{}_result.json'.format(d))
    with open(result_path, 'w') as f:
        json.dump(old_result, f, indent=4)

