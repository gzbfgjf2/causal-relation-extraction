from typing import List

import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast
)
from src.data_process_scripts_parts.read import (
    data_path_to_data_dict,
    data_paths_to_data_dict
)
from torch.utils.data import DataLoader, Dataset
from src.data_objects.dataset import CAUSAl_STATE
from src.metrics.metrics_token import (
    ExactMatchClassificationPrecision,
    ExactMatchClassificationRecall,
    ExactMatchClassificationF1,
    JaccardIndex
)
from src.experiment_scripts_parts.input import (
    mark_1d_tensor,
    data_paths_to_data_arr,
    # OneMapDataset,
    filter_data_dict,
    data_point_arr_to_causality_instances,
    data_point_to_batch_encoding, check_ids, check_train_test
)
from src.experiment_scripts_parts.input import select_by_ids
from src.models_main.connective_token_classification import (
    ConnectiveTokenClassification
)
from sklearn.model_selection import train_test_split
import pickle
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
            cause_label=None,
            effect_label=None
        )


class TestDataset(Dataset):
    def __init__(self, arr, tokenizer):
        self.data = arr
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def run_connective_token_classification(data_paths, epoch):
    data_dict = data_paths_to_data_dict(data_paths)
    filter_data_dict(data_dict)
    data_point_arr = list(data_dict.values())
    with open('dataset/full_ids', 'rb') as f:
        train_ids, val_ids, test_ids = pickle.load(f)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_data_point_arr = select_by_ids(data_point_arr, train_ids)
    val_data_point_arr = select_by_ids(data_point_arr, val_ids)
    test_data_point_arr = select_by_ids(data_point_arr, test_ids)
    check_train_test(train_data_point_arr, test_data_point_arr)
    train_arr = data_point_arr_to_causality_instances(train_data_point_arr)
    train_dataset = TrainDataset(
        train_arr,
        tokenizer,
    )
    val_dataset = TestDataset(
        val_data_point_arr,
        tokenizer
    )
    test_dataset = TestDataset(
        test_data_point_arr,
        tokenizer,
    )
    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=32,
        shuffle=True
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=32
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=32
    )
    model = ConnectiveTokenClassification(tokenizer)
    # model = ConnectiveTokenClassification.load_from_checkpoint(
    #     checkpoint_path=(
    #         "checkpoints/instance_tag_token"
    #         "/connective/epoch=4-step=3124.ckpt"
    #     ),
    #     tokenizer=tokenizer
    # )
    model_name = type(model).__name__

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epoch,
        default_root_dir='checkpoints/' + model_name,
        # callbacks=[EarlyStopping(monitor="val_f1", mode="max", patience=2)]
    )
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader
    )
    trainer.test(model=model, test_dataloaders=testloader)
    return
