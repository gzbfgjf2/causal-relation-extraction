import pickle
from itertools import zip_longest, permutations
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import torch
from transformers import BertTokenizerFast
from dataclasses import dataclass, field
from src.data_process_scripts_parts.read import (
    data_path_to_data_dict,
    data_paths_to_data_dict,
)
from src.data_objects.dataset import CAUSAl_STATE

from src.models_main.instance_token_classification import (
    InstanceTokenClassification,
)
import pathlib
from src.experiment_scripts_parts.input import filter_data_dict

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


def select_by_ids(arr, ids):
    res = [arr[x] for x in ids]
    return res


def run_instance_token_classification(data_paths, epoch):
    results_file = pathlib.Path("test_results")
    if results_file.exists():
        x = input("delete previous file? y:n ")
        if x == "y":
            print("previous results file deleted")
            results_file.unlink()
        else:
            assert False

    data_dict = data_paths_to_data_dict(data_paths)
    filter_data_dict(data_dict)
    data_point_arr = list(data_dict.values())
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    with open("dataset/full_ids", "rb") as f:
        train_ids, val_ids, test_ids = pickle.load(f)

    test_arr = select_by_ids(data_point_arr, test_ids)
    test_dataset = TestDataset(
        test_arr,
        tokenizer,
    )

    testloader = DataLoader(test_dataset, batch_size=None, num_workers=32)
    model = InstanceTokenClassification(tokenizer)
    model_name = type(model).__name__
    trainer = pl.Trainer(
        max_epochs=epoch, default_root_dir="checkpoints/" + model_name
    )
    trainer.test(model=model, test_dataloaders=testloader)
    return


def run_instance_real_data():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = InstanceTokenClassification(tokenizer)
    doc = []
    rows = [["sent", "cause", "connective", "effect"]]
    import csv

    with open("test/test-data.csv") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        next(csv_reader)
        for row in csv_reader:
            doc.append(row[2])
    for sentence in doc:
        rows.extend(model(sentence))
        print(sentence)
    with open("test/test-result.csv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
