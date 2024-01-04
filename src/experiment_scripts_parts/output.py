from src.data_objects.dataset import CausalityInstance, CausalityElement
import torch
from src.experiment_scripts_parts.input import mark_1d_tensor
from itertools import zip_longest, permutations
from src.metrics.metrics import tensor_to_jaccard


def extract_spans(sentence, spans):
    return [sentence[x:y] for x, y in spans]


def decoding_causality_instance(sentence, causal_instance):
    connective = extract_spans(sentence, causal_instance.connective.spans)
    cause = extract_spans(sentence, causal_instance.cause.spans)
    effect = extract_spans(sentence, causal_instance.effect.spans)
    return {'cause': cause, 'connective': connective, 'effect': effect}


class SpanProcessor:
    def __init__(self):
        self.res = []
        self.prev = float('-inf')

    def process(self, i, x, y):
        if i == self.prev + 1:
            self.res[-1][1] = y
        else:
            self.res.append([x, y])
        self.prev = i

    def get_tuple(self):
        return tuple(tuple(x) for x in self.res)


def tensor_to_causal_element(tensor, target, batch_encoding):
    if tensor is None: return CausalityElement(tuple())
    element = SpanProcessor()
    for i, token in enumerate(batch_encoding.tokens()):
        span = batch_encoding.token_to_chars(i)
        if not span: continue
        x, y = span
        if tensor[i] == target: element.process(i, x, y)
    return CausalityElement(element.get_tuple())


def tensor_to_causal_instance(tensor, batch_encoding):
    connective = tensor_to_causal_element(tensor, 1, batch_encoding)
    cause = tensor_to_causal_element(tensor, 2, batch_encoding)
    effect = tensor_to_causal_element(tensor, 3, batch_encoding)
    return CausalityInstance(
        connective=connective,
        cause=cause,
        effect=effect
    )


def tensor_to_contiguous_token_spans(tensor, target):
    res = set()
    start = -1
    for i, element in enumerate(tensor):
        if element.item() == target:
            if start == -1:
                start = i
        else:
            if start != -1:
                res.add((start, i))
                start = -1
    return res


def populate_targets(causality_instances, batch_encoding):
    res = []
    for k, causal_instance in causality_instances.items():
        connective_char_spans = causal_instance.connective.spans
        cause_char_spans = causal_instance.cause.spans
        effect_char_spans = causal_instance.effect.spans
        batch_encoding['labels'] = (
            torch.zeros_like(batch_encoding.input_ids[0])
        )
        for span, label in zip(
            (
                connective_char_spans,
                cause_char_spans,
                effect_char_spans
            ),
            (1, 2, 3)
        ):
            mark_1d_tensor(batch_encoding.labels, span, batch_encoding, label)
        res.append(batch_encoding['labels'].detach().clone())
    if not res:
        assert not causality_instances
        not_causal_target_tensor = torch.zeros_like(
            batch_encoding.input_ids[0]
        )
        assert not not_causal_target_tensor.requires_grad
        res.append(not_causal_target_tensor)
    return res


def model_output_to_1d_tensor(output, rg):
    pred = torch.argmax(output.logits, dim=-1)[0]
    pred[rg[0]:] = 0
    return pred


def targets_preds_to_best_pair_arrangement(targets, preds):
    best = -1
    best_situation = None
    if len(targets) > len(preds):
        situations = [
            list(zip_longest(x, preds)) for x in permutations(targets)
        ]
    else:
        situations = [
            list(zip_longest(targets, x)) for x in permutations(preds)
        ]
    for situation in situations:
        score = 0
        for x, y in situation:
            connective_j = tensor_to_jaccard(x, y, 1)
            score += connective_j
        if score > best:
            best_situation = situation
            best = score
    if not best_situation:
        assert False
    return best_situation
