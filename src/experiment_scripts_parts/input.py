from typing import List, Dict
from src.data_process_scripts_parts.read import (
    data_path_to_data_dict,
    data_paths_to_data_dict
)
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from src.data_objects.dataset import CAUSAl_STATE
import torch

import pickle

DIM = 256


def filter_data_dict(data_dict):
    to_be_removed = []
    for uid, data_point in data_dict.items():
        if data_point.is_causal == CAUSAl_STATE['not_determined']:
            to_be_removed.append(uid)
            continue
        causality_instances = data_point.causality_instances
        for (key, causality_instance) in tuple(causality_instances.items()):
            if len(causality_instance.connective.spans) != 1:
                del data_point.causality_instance[key]
    for k in to_be_removed:
        del data_dict[k]


def data_point_arr_to_causality_instances(data_point_arr) -> List:
    # [sentence, connective_span, cause_spans, effect_spans]
    causality_instances = []
    for data_point in data_point_arr:
        if data_point.is_causal == CAUSAl_STATE['not_determined']:
            assert False
        if data_point.is_causal == CAUSAl_STATE['not_causal']:
            causality_instances.append((data_point.sentence, None, None, None))
            continue
        casusality_instances = data_point.causality_instances
        for (
                connective_element,
                causality_instance
        ) in casusality_instances.items():
            if len(causality_instance.connective.spans) != 1: continue
            causality_instances.append((
                data_point.sentence,
                causality_instance.connective.spans,
                causality_instance.cause.spans,
                causality_instance.effect.spans
            ))
    return causality_instances


def data_paths_to_data_arr(
        data_paths,
        data_dict_to_arr
):
    return data_dict_to_arr(data_paths_to_data_dict(data_paths))


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
        tokenizer,
        connective_label=1,
        cause_label=2,
        effect_label=3
):
    (
        sentence,
        connective_spans,
        cause_spans,
        effect_spans
    ) = data_point

    batch_encoding = tokenizer(
        sentence,
        padding='max_length',
        max_length=DIM,
        truncation=True,
        return_tensors='pt'
    )

    for x, y in batch_encoding.items():
        if torch.is_tensor(y): batch_encoding[x] = y.squeeze()
    batch_encoding['labels'] = torch.zeros_like(batch_encoding.input_ids)
    if connective_spans is None:
        return batch_encoding
    assert connective_spans
    assert cause_spans
    assert effect_spans
    if connective_label is not None:
        mark_1d_tensor(
            batch_encoding.labels,
            connective_spans,
            batch_encoding,
            connective_label
        )
    if cause_label is not None:
        mark_1d_tensor(
            batch_encoding.labels,
            cause_spans,
            batch_encoding,
            cause_label
        )
    if effect_label is not None:
        mark_1d_tensor(
            batch_encoding.labels,
            effect_spans,
            batch_encoding,
            effect_label
        )
    return batch_encoding


def token_inside_spans(token_span, spans):
    x, y = token_span
    for nx, ny in spans:
        if nx <= x and y <= ny: return True
    return False


def focus_span_in_labels_spans(focus_span, labels_spans):
    for t in labels_spans:
        if focus_span == t: return True
    return False


def focus_span_is_correct(focus_span, labels_spans, batch_encoding):
    x = batch_encoding.token_to_chars(focus_span[0])
    if x is None: return False
    x = x[0]
    y = batch_encoding.token_to_chars(focus_span[1] - 1)
    if y is None: return False
    y = y[1]
    for nx, ny in labels_spans:
        if x == nx and y == ny: return True
    return False


def generate_training_point(
        focus_span,
        labels_spans,
        batch_encoding,
        target
):
    input_ids = batch_encoding.input_ids.detach().clone().numpy()
    attention_mask = batch_encoding.attention_mask.detach().clone().numpy()
    token_type_ids = torch.zeros_like(batch_encoding.input_ids).numpy()
    token_type_ids[focus_span[0]:focus_span[1]] = target
    # if focus_span_in_labels_spans(focus_span, labels_spans):
    if focus_span_is_correct(focus_span, labels_spans, batch_encoding):
        labels = 1
    else:
        labels = 0
    return input_ids, attention_mask, token_type_ids, labels


def concat(x, y):
    if x is None: return y
    return torch.cat((x, y), 0)


def generate_focus_spans(start, end, length):
    res = []
    # print(start, end, length)
    for x in range(start, end):
        for y in range(x + 1, min(x + 1 + length, end + 1)):
            res.append((x, y))
    return res


def char_spans_to_token_spans(char_spans, batch_encoding):
    res = []
    for x, y in char_spans:
        i = batch_encoding.char_to_token(x)
        # need to make char included so -1
        j = (
                batch_encoding.char_to_token(y - 1)
                or batch_encoding.char_to_token(y - 2)
        )
        if i is None or j is None:
            print(x, y, i, j)
            for i, x in enumerate(batch_encoding.tokens()):
                print(i)
                print(x)
                print(batch_encoding.token_to_chars(i))
            assert False
        # need to make j excluded, so +1
        res.append((i, j + 1))
    return res


def data_point_to_batch_encoding_single(data_point, tokenizer):
    sentence = data_point.sentence
    batch_encoding = tokenizer(
        sentence,
        padding='max_length',
        max_length=256,
        truncation=True,
        return_tensors='pt'
    )
    for x, y in batch_encoding.items():
        if torch.is_tensor(y): batch_encoding[x] = y.squeeze()
    batch_encoding['labels'] = torch.zeros_like(batch_encoding.input_ids)
    connective_spans = []
    for causality_instance in data_point.causality_instances.values():
        for spans in causality_instance.connective.spans:
            connective_spans.append(spans)
    mark_1d_tensor(
        batch_encoding.labels,
        connective_spans,
        batch_encoding,
        1
    )
    return batch_encoding


stats = [0, 0]


def data_point_to_batch_encoding_spans(data_point, tokenizer):
    sentence = data_point.sentence
    batch_encoding = tokenizer(
        sentence,
        padding='max_length',
        max_length=256,
        truncation=True,
        return_tensors='pt'
    )
    for x, y in batch_encoding.items():
        if torch.is_tensor(y): batch_encoding[x] = y.squeeze()
    start = 1
    end = torch.nonzero(
        batch_encoding.input_ids == 102,
        as_tuple=True
    )[0].item()
    length = 10
    focus_spans = generate_focus_spans(start, end, length)

    labels_char_spans = []
    for causality_instance in data_point.causality_instances.values():
        for span in causality_instance.connective.spans:
            labels_char_spans.append(span)

    # labels_spans = char_spans_to_token_spans(
    #     labels_char_spans,
    #     batch_encoding
    # )
    labels_spans = labels_char_spans
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    total_correct_focus_labels = 0
    for focus_span in focus_spans:
        (
            new_input_ids,
            new_attention_mask,
            new_token_type_ids,
            new_labels
        ) = generate_training_point(
            focus_span,
            labels_spans,
            batch_encoding,
            1
        )
        if new_labels == 1: total_correct_focus_labels += 1
        input_ids.append(new_input_ids)
        attention_mask.append(new_attention_mask)
        token_type_ids.append(new_token_type_ids)
        labels.append(new_labels)
    if total_correct_focus_labels != len(labels_char_spans):
        print(data_point)
        print(total_correct_focus_labels)
        print(len(labels_char_spans))
    else:
        stats[1] += 1
    stats[0] += 1
    print(stats)
    # assert False
    return input_ids, attention_mask, token_type_ids, labels


def check_ids(train_ids, test_ids):
    with open('train_ids', 'rb') as f:
        a = pickle.load(f)
    with open('test_ids', 'rb') as f:
        b = pickle.load(f)
    assert list(train_ids) == a and list(test_ids) == b


def check_train_test(train_arr, test_arr):
    a = set()
    for x in train_arr:
        a.add(x.sentence)
    assert len(a) == len(train_arr)
    for x in test_arr:
        if x.sentence in a: assert False


def select_by_ids(arr, ids):
    res = [arr[x] for x in ids]
    return res
