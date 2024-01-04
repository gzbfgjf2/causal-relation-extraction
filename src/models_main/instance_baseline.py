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
    mark_1d_tensor
)

from src.experiment_scripts_parts.output import (
    decoding_causality_instance,
    tensor_to_causal_instance,
    populate_targets,
    model_output_to_1d_tensor,
    targets_preds_to_best_pair_arrangement
)

from src.metrics.metrics import (
    calculate_metric, FullMetric, tensor_to_jaccard, spans_to_jaccard
)


DIM = 256


class Baseline(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        config = BertConfig(num_labels=4)
        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-uncased',
            config=config,
            ignore_mismatched_sizes=True
        )
        self.metric_1 = FullMetric.create(threshhold=1)
        self.metric_75 = FullMetric.create(threshhold=0.75)

    def basic_step(self, batch):
        # print(batch.labels)
        out = self.model(**batch)
        return out

    def forward(self, batch):
        return self.basic_step(batch)

    def training_step(self, batch, batch_idx):
        out = self.basic_step(batch)
        loss = out.loss
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch_encoding = self.tokenizer(
            batch.sentence,
            padding='max_length',
            max_length=DIM,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        targets = populate_targets(batch, batch_encoding)

        out = self.basic_step(batch_encoding)
        rg = torch.nonzero(batch_encoding.input_ids[0] == 102, as_tuple=True)
        pred = model_output_to_1d_tensor(out, rg)

        preds = [pred]

        if not targets or not preds:
            print(targets)
            print(preds)
            assert False

        best_situation = targets_preds_to_best_pair_arrangement(targets, preds)

        for x, y in best_situation:
            gold_instance = tensor_to_causal_instance(x, batch_encoding)
            preds_instance = tensor_to_causal_instance(y, batch_encoding)
            self.metric_1.update(gold_instance, preds_instance)
            self.metric_75.update(gold_instance, preds_instance)
            print(decoding_causality_instance(batch.sentence, gold_instance))
            print(decoding_causality_instance(batch.sentence, preds_instance))
        print('1:')
        print(self.metric_1.report())
        print('.75:')
        print(self.metric_75.report())
        print()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


# {'connective': {'jaccard': 0.7232401664532505, 'precision': 0.7912087912087912, 'recall': 0.75, 'f1': 0.770053475935829}, 'cause': {'jaccard': 0.6646325389321377, 'precision': 0.6021505376344086, 'recall': 0.7, 'f1': 0.6473988439306358}, 'effect': {'jaccard': 0.6202981413661783, 'precision': 0.5869565217391305, 'recall': 0.675, 'f1': 0.627906976744186}}
# .75:
# {'connective': {'jaccard': 0.7232401664532505, 'precision': 0.8351648351648352, 'recall': 0.76, 'f1': 0.7958115183246074}, 'cause': {'jaccard': 0.6646325389321377, 'precision': 0.7311827956989247, 'recall': 0.7391304347826086, 'f1': 0.7351351351351352}, 'effect': {'jaccard': 0.6202981413661783, 'precision': 0.7065217391304348, 'recall': 0.7142857142857143, 'f1': 0.7103825136612023}}
