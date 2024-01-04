from itertools import zip_longest, permutations
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import torch
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast
)
from dataclasses import dataclass, field

from src.data_process_scripts_parts.read import (
    data_path_to_data_dict,
    data_paths_to_data_dict
)

from src.experiment_scripts_parts.input import (
    filter_data_dict,
    data_point_arr_to_causality_instances,
    mark_1d_tensor,
    data_point_to_batch_encoding, check_ids, check_train_test, select_by_ids
)
import pickle
from src.experiment_scripts_parts.output import (
    decoding_causality_instance,
    tensor_to_causal_instance
)

from src.metrics.metrics import (
    calculate_metric, FullMetric, spans_to_jaccard
)

from src.models_main.instance_baseline import Baseline

DIM = 256


class TrainDataset(Dataset):
    def __init__(self, arr, tokenizer, data_point_to_training_point):
        self.data = arr
        self.tokenizer = tokenizer
        self.map = data_point_to_training_point

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.map(self.data[idx], self.tokenizer)


class TestDataset(Dataset):
    def __init__(self, arr, tokenizer):
        self.data = arr
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def run_instance_baseline(data_paths, epoch):
    data_dict = data_paths_to_data_dict(data_paths)
    filter_data_dict(data_dict)
    data_point_arr = list(data_dict.values())

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    with open('dataset/full_ids', 'rb') as f:
        train_ids, val_ids, test_ids = pickle.load(f)
    train_data_point_arr = select_by_ids(data_point_arr, train_ids)
    test_data_point_arr = select_by_ids(data_point_arr, test_ids)
    check_train_test(train_data_point_arr, test_data_point_arr)
    train_arr = data_point_arr_to_causality_instances(train_data_point_arr)

    train_dataset = TrainDataset(
        train_arr,
        tokenizer,
        data_point_to_batch_encoding
    )

    test_dataset = TestDataset(
        test_data_point_arr,
        tokenizer
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=32,
        shuffle=True
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=32
    )
    model = Baseline(tokenizer)
    model_name = type(model).__name__
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epoch,
        default_root_dir='checkpoints/' + model_name
    )
    trainer.fit(model=model, train_dataloaders=trainloader)
    trainer.test(model=model, test_dataloaders=testloader)

