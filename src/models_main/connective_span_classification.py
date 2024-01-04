from transformers import (
    BertConfig,
    BertForSequenceClassification
)
from src.data_objects.dataset import CAUSAl_STATE
from src.metrics.metrics_token import (
    ExactMatchClassificationPrecision,
    ExactMatchClassificationRecall,
    ExactMatchClassificationF1,
    JaccardIndex
)

from src.experiment_scripts_parts.input import (
    mark_1d_tensor,
    # data_paths_to_data_arr
)

from typing import List
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from transformers import BertTokenizerFast
import os
from src.experiment_scripts_parts.input import (
    mark_1d_tensor,
    data_paths_to_data_arr
)
from src.experiment_scripts_parts.input import (
    data_point_to_batch_encoding_spans,
    data_point_to_batch_encoding_single
)


class ConnectiveSpanClassification(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        config = BertConfig(num_labels=2)
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            config=config,
            ignore_mismatched_sizes=True
        )
        self.val_precision_metric = ExactMatchClassificationPrecision()
        self.val_recall = ExactMatchClassificationRecall()
        self.val_f1 = ExactMatchClassificationF1()
        self.val_jaccard = JaccardIndex()

        self.precision_metric = ExactMatchClassificationPrecision()
        self.recall = ExactMatchClassificationRecall()
        self.f1 = ExactMatchClassificationF1()
        self.jaccard = JaccardIndex()

    def basic_step(self, batch):
        out = self.model(**batch)
        return out

    def training_step(self, batch, batch_idx):
        out = self.basic_step(batch)
        loss = out.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            input_ids,
            attention_mask,
            token_type_ids,
            labels
        ) = data_point_to_batch_encoding_spans(
            batch,
            self.tokenizer
        )
        input_ids = torch.tensor(
            input_ids,
            requires_grad=False,
            device=self.device
        )
        attention_mask = torch.tensor(
            attention_mask,
            requires_grad=False,
            device=self.device
        )
        token_type_ids = torch.tensor(
            token_type_ids,
            requires_grad=False,
            device=self.device
        )
        labels = torch.tensor(
            labels,
            requires_grad=False,
            device=self.device
        )
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        preds = torch.argmax(out.logits, dim=-1)
        preds_idx = torch.nonzero(preds == 1, as_tuple=True)
        preds = token_type_ids[preds_idx]
        preds_1d = torch.sum(preds, dim=0)
        preds_1d[torch.nonzero(preds_1d != 0, as_tuple=True)] = 1

        batch_encoding = data_point_to_batch_encoding_single(batch,
                                                             self.tokenizer)
        labels_1d = batch_encoding['labels']

        self.val_precision_metric(labels_1d, preds_1d, 1)
        self.val_recall(labels_1d, preds_1d, 1)
        self.val_f1(labels_1d, preds_1d, 1)
        self.val_jaccard(labels_1d, preds_1d, 1)
        self.log("val_precision_step", self.val_precision_metric, on_epoch=True)
        self.log("val_recall_step", self.val_recall, on_epoch=True)
        self.log("val_f1_step", self.val_f1, on_epoch=True)
        self.log("val_jaccard", self.val_jaccard, on_epoch=True)
        return out.loss

    def test_step(self, batch, batch_idx):
        (
            input_ids,
            attention_mask,
            token_type_ids,
            labels
        ) = data_point_to_batch_encoding_spans(
            batch,
            self.tokenizer
        )
        input_ids = torch.tensor(
            input_ids,
            requires_grad=False,
            device=self.device
        )
        attention_mask = torch.tensor(
            attention_mask,
            requires_grad=False,
            device=self.device
        )
        token_type_ids = torch.tensor(
            token_type_ids,
            requires_grad=False,
            device=self.device
        )
        labels = torch.tensor(
            labels,
            requires_grad=False,
            device=self.device
        )
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        preds = torch.argmax(out.logits, dim=-1)
        preds_idx = torch.nonzero(preds == 1, as_tuple=True)
        preds = token_type_ids[preds_idx]
        preds_1d = torch.sum(preds, dim=0)
        preds_1d[torch.nonzero(preds_1d != 0, as_tuple=True)] = 1

        batch_encoding = data_point_to_batch_encoding_single(batch,
                                                             self.tokenizer)
        labels_1d = batch_encoding['labels']

        self.precision_metric(labels_1d, preds_1d, 1)
        self.recall(labels_1d, preds_1d, 1)
        self.f1(labels_1d, preds_1d, 1)
        self.jaccard(labels_1d, preds_1d, 1)
        self.log("test_precision_step", self.precision_metric, on_epoch=True)
        self.log("test_recall_step", self.recall, on_epoch=True)
        self.log("test_f1_step", self.f1, on_epoch=True)
        self.log("jaccard", self.jaccard, on_epoch=True)
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


