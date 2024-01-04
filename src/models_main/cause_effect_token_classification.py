import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast
)

from src.metrics.metrics_token import CauseEffectConfusionPrecision, \
    CauseEffectConfusionRecall, CauseEffectConfusionF1, CEJaccardIndex
from src.metrics.metrics import MetricElement
from src.experiment_scripts_parts.output import (
    populate_targets,
    model_output_to_1d_tensor,
    tensor_to_contiguous_token_spans,
    targets_preds_to_best_pair_arrangement,
    tensor_to_causal_instance,
    decoding_causality_instance
)

from transformers import BertConfig, BertForTokenClassification


class CauseEffectTokenClassification(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        config = BertConfig(num_labels=3)
        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-uncased',
            config=config,
            ignore_mismatched_sizes=True
        )
        self.cause_precision_metric = CauseEffectConfusionPrecision(1)
        self.cause_recall = CauseEffectConfusionRecall(1)
        self.cause_f1 = CauseEffectConfusionF1(1)
        self.cause_jaccard = CEJaccardIndex()
        self.effect_precision_metric = CauseEffectConfusionPrecision(1)
        self.effect_recall = CauseEffectConfusionRecall(1)
        self.effect_f1 = CauseEffectConfusionF1(1)
        self.effect_jaccard = CEJaccardIndex()

        self.sents = set()

    def basic_step(self, batch):
        out = self.model(**batch)
        return out

    def forward(self, batch):
        return self.basic_step(batch)

    def training_step(self, batch, batch_idx):

        batch, sent = batch
        self.sents.add(sent)
        out = self.basic_step(batch)
        loss = out.loss
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch, sent = batch
        if sent in self.sents: assert False
        out = self.basic_step(batch)
        loss = out.loss
        labels = batch['labels']
        rg = torch.nonzero(batch['input_ids'][0] == 102, as_tuple=True)
        labels_1d = labels[0]
        preds = torch.argmax(out['logits'], dim=-1)
        preds_1d = preds[0]
        preds_1d[rg[0]:] = 0

        self.cause_precision_metric(labels_1d, preds_1d, 1)
        self.cause_recall(labels_1d, preds_1d, 1)
        self.cause_f1(labels_1d, preds_1d, 1)
        self.cause_jaccard(labels_1d, preds_1d, 1)

        self.effect_precision_metric(labels_1d, preds_1d, 2)
        self.effect_recall(labels_1d, preds_1d, 2)
        self.effect_f1(labels_1d, preds_1d, 2)
        self.effect_jaccard(labels_1d, preds_1d, 2)

        self.log(
            "test_cause_precision_step",
            self.cause_precision_metric,
            on_epoch=True
        )
        self.log(
            "test_cause_recall_step",
            self.cause_recall,
            on_epoch=True
        )
        self.log(
            "test_cause_f1_step",
            self.cause_f1,
            on_epoch=True
        )
        self.log(
            "cause_jaccard",
            self.cause_jaccard,
            on_epoch=True
        )
        self.log(
            "test_effect_precision_step",
            self.effect_precision_metric,
            on_epoch=True
        )
        self.log(
            "test_effect_recall_step",
            self.effect_recall,
            on_epoch=True
        )
        self.log(
            "test_effect_f1_step",
            self.effect_f1,
            on_epoch=True
        )
        self.log(
            "effect_jaccard",
            self.effect_jaccard,
            on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
