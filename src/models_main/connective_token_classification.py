import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast
)
from src.metrics.metrics import MetricElement
from src.experiment_scripts_parts.output import (
    populate_targets,
    model_output_to_1d_tensor,
    tensor_to_contiguous_token_spans,
    targets_preds_to_best_pair_arrangement,
    tensor_to_causal_instance,
    decoding_causality_instance
)


class ConnectiveTokenClassification(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        config = BertConfig(num_labels=2)
        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-uncased',
            config=config,
            ignore_mismatched_sizes=True
        )
        self.metric = MetricElement()
        self.val_metric = MetricElement()

    def basic_step(self, batch):
        out = self.model(**batch)
        return out

    def forward(self, batch):
        return self.basic_step(batch)

    def training_step(self, batch, batch_idx):
        out = self.basic_step(batch)
        loss = out.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_encoding = self.tokenizer(
            batch.sentence,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        best_situation = batch_encoding_to_best_situation(
            self,
            batch_encoding,
            batch.causality_instances
        )
        for x, y in best_situation:
            gold_instance = tensor_to_causal_instance(x, batch_encoding)
            preds_instance = tensor_to_causal_instance(y, batch_encoding)
            self.val_metric.update(
                gold_instance.connective.spans,
                preds_instance.connective.spans
            )
        val_f1 = self.val_metric.compute()['f1']
        self.log('val_f1', val_f1)
        print(val_f1)
        return val_f1

    def test_step(self, batch, batch_idx):
        batch_encoding = self.tokenizer(
            batch.sentence,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        best_situation = batch_encoding_to_best_situation(
            self,
            batch_encoding,
            batch.causality_instances
        )
        evaluation_best_situation(
            best_situation,
            batch_encoding,
            batch.sentence,
            self.metric
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


def batch_encoding_to_best_situation(self, batch_encoding, causality_instances):
    targets = populate_targets(causality_instances, batch_encoding)
    if 'labels' in batch_encoding: del batch_encoding['labels']
    out = self.basic_step(batch_encoding)
    rg = torch.nonzero(batch_encoding.input_ids[0] == 102, as_tuple=True)
    connective_pred = model_output_to_1d_tensor(out, rg)

    for x, y in batch_encoding.items():
        if torch.is_tensor(y): batch_encoding[x] = y.squeeze()
    preds = []
    connective_spans = tensor_to_contiguous_token_spans(connective_pred, 1)
    for connective_start, connective_end in connective_spans:
        token_type_ids = torch.zeros_like(
            batch_encoding.input_ids,
            requires_grad=False
        )
        # print(batch_encoding.tokens()[connective_start:connective_end])
        token_type_ids[connective_start:connective_end] = 1
        token_type_ids[rg[0]:] = 0
        preds.append(token_type_ids)

    best_situation = targets_preds_to_best_pair_arrangement(targets, preds)
    return best_situation


def evaluation_best_situation(
        best_situation,
        batch_encoding,
        sentence,
        metric
):
    for x, y in best_situation:
        gold_instance = tensor_to_causal_instance(x, batch_encoding)
        preds_instance = tensor_to_causal_instance(y, batch_encoding)
        metric.update(
            gold_instance.connective.spans,
            preds_instance.connective.spans
        )
        print('gold')
        print(decoding_causality_instance(sentence, gold_instance))
        print('preds')
        print(decoding_causality_instance(sentence, preds_instance))
    print('1:')
    print(metric.compute())
    print()
    return
