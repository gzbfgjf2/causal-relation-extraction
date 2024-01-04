import sys
import csv
import pytorch_lightning as pl

from src.experiment_scripts_main.connective_token_classification import (
    ConnectiveTokenClassification
)
from src.experiment_scripts_main.cause_effect_token_tagging import (
    CauseEffectTokenClassification
)

from src.metrics.metrics import (
    FullMetric
)
import torch

from src.experiment_scripts_parts.output import (
    tensor_to_contiguous_token_spans,
    tensor_to_causal_instance,
    decoding_causality_instance,
    populate_targets,
    model_output_to_1d_tensor,
    targets_preds_to_best_pair_arrangement
)

DIM = 256


class InstanceTokenClassification(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.connective = (
            ConnectiveTokenClassification
                .load_from_checkpoint(
                checkpoint_path=(
                    "checkpoints/instance_tag_token"
                    "/connective/epoch=2-step=3491.ckpt"
                ),
                tokenizer=tokenizer
            )
        )
        self.cause_effect = (
            CauseEffectTokenClassification
                .load_from_checkpoint(
                checkpoint_path=(
                    "checkpoints/instance_tag_token/cause_effect"
                    "/epoch=2-step=2765.ckpt"
                ),
                tokenizer=tokenizer
            )
        )

        self.metric_1 = FullMetric.create(threshhold=1)
        self.metric_75 = FullMetric.create(threshhold=0.75)

    def basic_step(self, batch):
        out = self.model(**batch)
        return out

    def forward(self, sentence):
        batch_encoding = self.tokenizer(
            sentence,
            padding='max_length',
            max_length=DIM,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        if 'labels' in batch_encoding: del batch_encoding['labels']
        connective_out = self.connective(batch_encoding)
        # rg = torch.nonzero(batch_encoding.input_ids[0] == 102, as_tuple=True)
        connective_pred = torch.argmax(connective_out.logits, dim=-1)[0]
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
            ce_input = {
                'input_ids': batch_encoding['input_ids'].unsqueeze(0),
                'attention_mask': (
                    batch_encoding['attention_mask'].unsqueeze(0)
                ),
                'token_type_ids': token_type_ids.unsqueeze(0)
            }
            cause_effect_out = self.cause_effect(ce_input)
            ce_preds = torch.argmax(cause_effect_out.logits, dim=-1)[0]
            token_type_ids[torch.nonzero(ce_preds == 1, as_tuple=True)] = 2
            token_type_ids[torch.nonzero(ce_preds == 2, as_tuple=True)] = 3
            # token_type_ids[rg[0]:] = 0
            preds.append(token_type_ids)

        # best_situation = targets_preds_to_best_pair_arrangement(targets, preds)
        res_arr = []
        for y in preds:
            preds_instance = tensor_to_causal_instance(y, batch_encoding)
            res = decoding_causality_instance(sentence, preds_instance)
            curr =[
                sentence, 
                res['cause'],
                res['connective'],
                res['effect']
            ]
            res_arr.append(curr)
        if not res_arr:
            res_arr.append([sentence, '', '', ''])
        return res_arr

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
        targets = populate_targets(batch.causality_instances, batch_encoding)
        if 'labels' in batch_encoding: del batch_encoding['labels']
        connective_out = self.connective(batch_encoding)
        rg = torch.nonzero(batch_encoding.input_ids[0] == 102, as_tuple=True)
        connective_pred = model_output_to_1d_tensor(connective_out, rg)
        for x, y in batch_encoding.items():
            if torch.is_tensor(y): batch_encoding[x] = y.squeeze()
        preds = []
        connective_spans = tensor_to_contiguous_token_spans(connective_pred, 1)
        for connective_start, connective_end in connective_spans:
            token_type_ids = torch.zeros_like(
                batch_encoding.input_ids,
                requires_grad=False
            )
            print(batch_encoding.tokens()[connective_start:connective_end])
            token_type_ids[connective_start:connective_end] = 1
            ce_input = {
                'input_ids': batch_encoding['input_ids'].unsqueeze(0),
                'attention_mask': (
                    batch_encoding['attention_mask'].unsqueeze(0)
                ),
                'token_type_ids': token_type_ids.unsqueeze(0)
            }
            cause_effect_out = self.cause_effect(ce_input)
            ce_preds = torch.argmax(cause_effect_out.logits, dim=-1)[0]
            token_type_ids[torch.nonzero(ce_preds == 1, as_tuple=True)] = 2
            token_type_ids[torch.nonzero(ce_preds == 2, as_tuple=True)] = 3
            token_type_ids[rg[0]:] = 0
            preds.append(token_type_ids)

        best_situation = targets_preds_to_best_pair_arrangement(targets, preds)

        with open('test_results', 'a') as f:
            f.write(batch.sentence + '\n')
            for x, y in best_situation:
                gold_instance = tensor_to_causal_instance(x, batch_encoding)
                preds_instance = tensor_to_causal_instance(y, batch_encoding)
                self.metric_1.update(gold_instance, preds_instance)
                self.metric_75.update(gold_instance, preds_instance)
                gold = (
                    decoding_causality_instance(
                        batch.sentence,
                        gold_instance
                    )
                )
                print(gold)
                f.write('gold: \n')
                for k, v in gold.items():
                    f.write(f'{k}\n')
                    f.write(f'{v}\n')
                preds = (
                    decoding_causality_instance(
                        batch.sentence,
                        preds_instance
                    )
                )
                print(preds)
                f.write('preds: \n')
                for k, v in preds.items():
                    f.write(f'{k}\n')
                    f.write(f'{v}\n')
                print()
                f.write('\n\n')
            print('1:')
            print(self.metric_1.report())
            print('.75:')
            print(self.metric_75.report())
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
