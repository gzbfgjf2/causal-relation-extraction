
from torchmetrics import Metric
import torch
from itertools import permutations, zip_longest
from src.experiment_scripts_parts.output import tensor_to_contiguous_token_spans


class ExactMatchClassificationBase(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('tp', default=torch.tensor(0, requires_grad=False))
        self.add_state('fp', default=torch.tensor(0, requires_grad=False))
        self.add_state('fn', default=torch.tensor(0, requires_grad=False))

    def update(self, labels_1d, preds_1d, target):
        labels_spans = tensor_to_contiguous_token_spans(labels_1d, target)
        preds_spans = tensor_to_contiguous_token_spans(preds_1d, target)
        tp = torch.tensor(
            len(labels_spans.intersection(preds_spans)),
            requires_grad=False
        )
        fp = torch.tensor(len(preds_spans), requires_grad=False) - tp
        fn = torch.tensor(len(labels_spans), requires_grad=False) - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn


class ExactMatchClassificationPrecision(
    ExactMatchClassificationBase
):
    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        return precision


class ExactMatchClassificationRecall(
    ExactMatchClassificationBase
):
    def compute(self):
        recall = self.tp / (self.tp + self.fn)
        return recall


class ExactMatchClassificationF1(
    ExactMatchClassificationBase
):
    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1


def two_spans_to_jaccard_index(x, y):
    if x is None or y is None: return 0
    x_start, x_end = x
    y_start, y_end = y
    if y_end <= x_start or x_end <= y_start: return 0
    intersection = min(x_end, y_end) - max(x_start, y_start)
    union = x_end - x_start + y_end - y_start - intersection
    return intersection / union


class JaccardIndex(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('index', default=torch.tensor(0, dtype=torch.float))
        self.add_state('total', default=torch.tensor(0))

    def update(self, labels_1d, preds_1d, target):
        a = list(tensor_to_contiguous_token_spans(labels_1d, target))
        b = list(tensor_to_contiguous_token_spans(preds_1d, target))
        print(a)
        print(b)
        best = 0
        if len(a) > len(b):
            situations = [list(zip_longest(x, b)) for x in permutations(a)]
        else:
            situations = [list(zip_longest(a, x)) for x in permutations(b)]
        for situation in situations:
            tmp = 0
            for match_pair in situation:
                x, y = match_pair
                tmp += two_spans_to_jaccard_index(x, y)
            best = max(best, tmp)
        self.index += torch.tensor(best)
        self.total += torch.tensor(len(a))

        # labels_binary = labels_1d == target
        # preds_binary = preds_1d == target
        # intersection = torch.bitwise_and(labels_binary, preds_binary)
        # below does not work
        # seems that a==b==c happens simutaneously rather than
        # (a==b)==c
        # intersection = labels_1d==target == (preds_1d==target)
        # union = torch.bitwise_or(labels_binary, preds_binary)
        # self.index += torch.sum(intersection)/torch.sum(union)
        # self.total += torch.tensor(1)
        # print(id(self))
        # print('uindex', self.index, id(self.index))
        # print('utotal', self.total)
        # print('ujaccard', self.index/self.total)

    def compute(self):
        # print('index', self.index)
        # print('total', self.total)
        # print('jaccard', self.index/self.total)
        return self.index / self.total


class CauseEffectConfusionBase(Metric):

    def __init__(self, thresh_hold):
        super().__init__()
        self.add_state('tp', default=torch.tensor(0, requires_grad=False))
        self.add_state('fp', default=torch.tensor(0, requires_grad=False))
        self.add_state('fn', default=torch.tensor(0, requires_grad=False))
        self.th = thresh_hold

    def update(self, labels_1d, preds_1d, target):
        labels_binary = labels_1d == target
        preds_binary = preds_1d == target
        intersection = torch.bitwise_and(labels_binary, preds_binary)
        intersection_length = torch.sum(intersection)
        union_length = (
                torch.sum(labels_binary)
                + torch.sum(preds_binary)
                - intersection_length
        )
        jaccard_index = intersection_length / union_length
        if jaccard_index >= self.th:
            print(labels_1d, preds_1d)
            self.tp += torch.tensor(1)
        elif torch.sum(labels_binary) > 0 and torch.sum(preds_binary) == 0:
            self.fn += torch.tensor(1)
        else:
            self.fp += torch.tensor(1)


class CauseEffectConfusionPrecision(
    CauseEffectConfusionBase
):
    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        return precision


class CauseEffectConfusionRecall(
    CauseEffectConfusionBase
):
    def compute(self):
        recall = self.tp / (self.tp + self.fn)
        return recall


class CauseEffectConfusionF1(
    CauseEffectConfusionBase
):
    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        print(precision)
        print(recall)
        f1 = 2 * precision * recall / (precision + recall)
        print(f1)
        return f1


class CEJaccardIndex(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('index', default=torch.tensor(0, dtype=torch.float))
        self.add_state('total', default=torch.tensor(0))

    def update(self, labels_1d, preds_1d, target):
        labels_binary = labels_1d == target
        preds_binary = preds_1d == target
        intersection = torch.bitwise_and(labels_binary, preds_binary)
        intersection_length = torch.sum(intersection)
        union_length = (
                torch.sum(labels_binary)
                + torch.sum(preds_binary)
                - intersection_length
        )
        jaccard_index = intersection_length / union_length
        self.index += torch.tensor(jaccard_index.item())
        self.total += torch.tensor(1)

    def compute(self):
        return self.index / self.total
