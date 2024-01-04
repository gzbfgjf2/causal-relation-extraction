import pickle
from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizerFast
)

from src.data_objects.dataset import CAUSAl_STATE
from src.data_process_scripts_parts.read import data_paths_to_data_dict
from src.experiment_scripts_parts.input import (
    filter_data_dict,
    check_train_test
)
from src.experiment_scripts_parts.input import select_by_ids
from src.models_main.cause_effect_token_classification import (
    CauseEffectTokenClassification
)


def data_point_arr_to_causality_instances(data_point_arr) -> List:
    # [sentence, connective_span, cause_spans, effect_spans]
    res = []
    for data_point in data_point_arr:
        if data_point.is_causal == CAUSAl_STATE['not_determined']:
            assert False
        if data_point.is_causal == CAUSAl_STATE['not_causal']:
            # causality_instances.append((data_point.sentence, None, None, None))
            continue
        causality_instances = data_point.causality_instances
        for (
                connective_element,
                causality_instance
        ) in causality_instances.items():
            if len(causality_instance.connective.spans) != 1: continue
            res.append((
                data_point.sentence,
                causality_instance.connective.spans,
                causality_instance.cause.spans,
                causality_instance.effect.spans
            ))
    return res


def token_inside_spans(token_span, spans):
    x, y = token_span
    for nx, ny in spans:
        if nx <= x and y <= ny: return True
        # if nx == x or y == ny: return True
    return False


def mark_1d_tensor(tensor, char_spans, batch_encoding, label_idx):
    # for each token, convert them to character spans, then
    # check if it is in labels' character span
    for i, token in enumerate(batch_encoding.tokens()):
        if token == '[PAD]': break
        token_char_span = batch_encoding.token_to_chars(i)
        if token_char_span is None: continue
        if token_inside_spans(token_char_span, char_spans):
            # if token_char_span[0] == char_spans[0]
            tensor[i] = label_idx
    return tensor


def data_point_to_batch_encoding(
        data_point,
        tokenizer
):
    sentence = data_point[0]
    batch_encoding = tokenizer(
        sentence,
        padding='max_length',
        max_length=256,
        truncation=True,
        return_tensors='pt'
    )
    for x, y in batch_encoding.items():
        if torch.is_tensor(y): batch_encoding[x] = y.squeeze()
    batch_encoding['token_type_ids'] = (
        torch.zeros_like(batch_encoding.input_ids)
    )
    batch_encoding['labels'] = torch.zeros_like(batch_encoding.input_ids)
    mark_1d_tensor(
        batch_encoding.token_type_ids,
        data_point[1],
        batch_encoding,
        1
    )
    mark_1d_tensor(
        batch_encoding.labels,
        data_point[2],
        batch_encoding,
        1
    )
    mark_1d_tensor(
        batch_encoding.labels,
        data_point[3],
        batch_encoding,
        2
    )
    return batch_encoding, sentence


class TrainDataset(Dataset):
    def __init__(self, arr, tokenizer):
        self.data = arr
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return data_point_to_batch_encoding(
            self.data[idx],
            self.tokenizer,
        )


class TestDataset(Dataset):
    def __init__(self, arr, tokenizer):
        self.data = arr
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def run_cause_effect_token_classification(data_paths, epoch):
    data_dict = data_paths_to_data_dict(data_paths)
    filter_data_dict(data_dict)
    data_point_arr = list(data_dict.values())
    with open('dataset/full_ids', 'rb') as f:
        train_ids, val_ids, test_ids = pickle.load(f)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_data_point_arr = select_by_ids(data_point_arr, train_ids)
    test_data_point_arr = select_by_ids(data_point_arr, test_ids)
    check_train_test(train_data_point_arr, test_data_point_arr)
    train_arr = data_point_arr_to_causality_instances(
        train_data_point_arr)
    test_arr = data_point_arr_to_causality_instances(
        test_data_point_arr)

    train_dataset = TrainDataset(
        train_arr,
        tokenizer,
    )

    test_dataset = TrainDataset(
        test_arr,
        tokenizer,
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=32,
        shuffle=True
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=32
    )
    model = CauseEffectTokenClassification(tokenizer)
    model_name = type(model).__name__
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epoch,
        default_root_dir='checkpoints/' + model_name
    )
    trainer.fit(model=model, train_dataloaders=trainloader)
    trainer.test(model=model, test_dataloaders=testloader)
