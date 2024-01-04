from typing import List, Dict
import jsonpickle
from src.data_objects.dataset import DataPoint
from src.data_objects.dataset import (
    DataPoint,
    CausalityInstance,
    CausalityElement
)


def data_path_to_data_dict(data_path):
    with open(data_path, 'r') as f: json_pickle_str = f.read()
    data_dict = jsonpickle.decode(json_pickle_str)
    return data_dict


def data_paths_to_data_dict(data_paths: List) -> Dict:
    data_dict = {}
    for data_path in data_paths:
        data_dict = data_dict | data_path_to_data_dict(data_path)
    return data_dict


    
