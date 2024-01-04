import pickle

# from transformers import (
#     BertConfig,
#     BertForSequenceClassification
# )
from src.data_objects.dataset import CAUSAl_STATE
from src.data_process_scripts_parts.read import data_paths_to_data_dict
# from src.metrics.metrics_token import (
#     ExactMatchClassificationPrecision,
#     ExactMatchClassificationRecall,
#     ExactMatchClassificationF1,
#     JaccardIndex
# )

from src.experiment_scripts_parts.input import (
    mark_1d_tensor, filter_data_dict, data_point_to_batch_encoding_spans,
    # data_paths_to_data_arr
)

from typing import List
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from transformers import BertTokenizerFast
import os

from src.models_main.connective_span_classification import \
    ConnectiveSpanClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def data_dict_to_data_points_with_contiguous_connectives(data_dict) -> List:
    data_points_with_contiguous_connectives = []
    for uid, data_point in data_dict.items():
        if data_point.is_causal == CAUSAl_STATE['not_determined']:
            assert data_point.latest_update_time == ''
            continue
        causality_instances = data_point.causality_instances
        # if data_point is certain it is not causal: then add the
        # sentence and empty connective spans
        if data_point.is_causal == CAUSAl_STATE['not_causal']:
            assert len(data_point.causality_instances) == 0
            data_points_with_contiguous_connectives.append(data_point)
            continue
        all_continuous = True
        for connective_key, causality_instance in (
                causality_instances.items()
        ):
            if len(causality_instance.connective.spans) != 1:
                all_continuous = False
                break
        if not all_continuous: continue
        data_points_with_contiguous_connectives.append(data_point)
    return data_points_with_contiguous_connectives


# data point:
#     sentence
#     conne 


class TrainDataset(Dataset):
    def __init__(self, arr):
        self.data = arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = {}
        for i, key in enumerate((
                'input_ids',
                'attention_mask',
                'token_type_ids',
                'labels'
        )):
            res[key] = self.data[idx][i]
        return res


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


def flatten_train_arr(arr, tokenizer):
    x = [[] for _ in range(4)]
    for element in arr:
        item = data_point_to_batch_encoding_spans(element, tokenizer)
        for i in range(4):
            x[i].extend(item[i])
    x = list(zip(*x))
    return x


def run_connective_span_classification(data_paths, epoch):
    data_dict = data_paths_to_data_dict(data_paths)
    filter_data_dict(data_dict)
    data_point_arr = list(data_dict.values())
    with open('dataset/full_ids', 'rb') as f:
        train_ids, val_ids, test_ids = pickle.load(f)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_data_point_arr = select_by_ids(data_point_arr, train_ids)
    val_data_point_arr = select_by_ids(data_point_arr, val_ids)
    test_data_point_arr = select_by_ids(data_point_arr, test_ids)
    train_arr = flatten_train_arr(train_data_point_arr, tokenizer)
    trues = 0
    print(len(train_arr))
    for x in train_arr:
        if x[3] == 1: trues += 1

    mul = len(train_arr) // trues
    # print(mul)
    # print(trues)
    # print(mul*trues+len(tr))
    over_sample = []
    for x in train_arr:
        if x[3] == 1: over_sample.extend([x] * mul)
    train_arr.extend(over_sample)
    # print(len(tr))
    tr = TrainDataset(train_arr)
    trainloader = DataLoader(
        tr,
        batch_size=32,
        shuffle=True,
        num_workers=32
    )
    model = ConnectiveSpanClassification(tokenizer)
    model_name = type(model).__name__
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epoch,
        default_root_dir='checkpoints/' + model_name
    )
    val_dataset = TestDataset(val_data_point_arr, tokenizer)
    test_dataset = TestDataset(test_data_point_arr, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=None, num_workers=1)
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=val_loader
    )
    trainer.test(model=model, test_dataloaders=test_loader)

