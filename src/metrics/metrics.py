from torchmetrics import Metric
import torch
from itertools import permutations, zip_longest
from dataclasses import dataclass, field


def spans_to_jaccard(labels_spans, preds_spans):
    labels_arr = list(labels_spans)
    preds_arr = list(preds_spans)

    union_and_intersection = 0
    intersection = 0
    i = j = 0
    while i<len(labels_arr) and j<len(preds_arr):
        x, y = labels_arr[i]
        m, n = preds_arr[j] 
        if x<=n and m<=y:
            intersection += min(y, n) - max(x, m)
        # union_and_intersection += y-x+m-n
        if y <= n: i += 1
        elif n <= y: j += 1
    
    for x, y in labels_arr:
        union_and_intersection += y-x
    for x, y in preds_arr:
        union_and_intersection += y-x
    union = union_and_intersection - intersection
    assert union >= 0
    assert intersection >= 0
    if union == 0:
        assert intersection == 0
        return 0
    return intersection/union
        

def tensor_to_jaccard(labels_1d, preds_1d, target):
    if labels_1d is None or preds_1d is None: return 0
    labels_binary = labels_1d == target
    preds_binary = preds_1d == target
    intersection = torch.bitwise_and(labels_binary, preds_binary)
    intersection_length = torch.sum(intersection)
    union_length = (
            torch.sum(labels_binary)
            + torch.sum(preds_binary)
            - intersection_length
    )
    if union_length.item() == 0:
        assert intersection_length == 0
        return 0
    jaccard_index = intersection_length / union_length
    return jaccard_index.item()


def calculate_metric(x, y, threshhold):
    # print(x, y)
    if not x and not y: return None
    if not x:
        return {
            'jaccard': 0,
            'tp': 0,
            'fp': 1,
            'fn': 0
        }
    elif not y:
        # print('y is tuple')
        return {
            'jaccard': 0,
            'tp': 0,
            'fp': 0,
            'fn': 1
        }
    else:
        jaccard = spans_to_jaccard(x, y)
        if jaccard >= threshhold:
            tp = 1
            fp = 0
        else:
            tp = 0
            fp = 1
        return {
            'jaccard': jaccard,
            'tp': tp,
            'fp': fp,
            'fn': 0
        }


@dataclass
class MetricElement:
    jaccard: int = 0
    total: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    threshhold: int = 1

    def compute(self):
        jac = self.jaccard / self.total if self.total else 0
        if self.tp + self.fp != 0:
            precision = self.tp / (self.tp + self.fp)
        else:
            precision = 0
        if self.tp + self.fn != 0:
            recall = self.tp / (self.tp + self.fn)
        else:
            recall = 0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return {
            'jaccard': jac,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _update(self, metric):
        self.jaccard += metric['jaccard']
        self.total += 1
        self.tp += metric['tp']
        self.fp += metric['fp']
        self.fn += metric['fn']
   
    def update(self, x, y):
        metric = calculate_metric(x, y, self.threshhold)
        if metric is None: return
        self._update(metric)


@dataclass
class FullMetric:
    connective: MetricElement 
    cause: MetricElement 
    effect: MetricElement

    @classmethod
    def create(cls, threshhold):
        connective = MetricElement(threshhold=threshhold)
        cause = MetricElement(threshhold=threshhold)
        effect = MetricElement(threshhold=threshhold)
        return cls(connective=connective, cause=cause, effect=effect)

    def report(self):
        # self.connective.compute()
        # print(self.connective.tp)
        # print(self.connective.fp)
        # print(self.connective.fn)
        # print(self.connective.total)
        return {
            'connective': self.connective.compute(),
            'cause': self.cause.compute(),
            'effect': self.effect.compute()
        }

    def update(self, gold_instance, preds_instance):
        # print(gold_instance, preds_instance)
        self.connective.update(
            gold_instance.connective.spans,
            preds_instance.connective.spans
        )
        self.cause.update(
            gold_instance.cause.spans,
            preds_instance.cause.spans
        )
        self.effect.update(
            gold_instance.effect.spans,
            preds_instance.effect.spans
        )

