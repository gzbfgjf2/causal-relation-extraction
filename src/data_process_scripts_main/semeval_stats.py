import copy
from itertools import zip_longest, permutations

import jsonpickle

from src.data_objects.dataset import CausalityInstance
from src.data_process_scripts_parts.read import (
    data_path_to_data_dict
)
from pathlib import Path

from src.metrics.metrics import spans_to_jaccard, FullMetric

folder = Path.home() / 'work/data/semeval/'

a = data_path_to_data_dict(folder / 'self-labeled-1-600/data')
b = data_path_to_data_dict(folder / 'self-labeled-400-1000/data')


def two_hundred(data_dict):
    res = {}
    for k, v in data_dict.items():
        if isinstance(v.causality_id, str): continue
        if 399 <= v.causality_id <= 599:
            res[k] = v
    return res


a_200 = two_hundred(a)
b_200 = two_hundred(b)
print(len(a_200))
print(len(b_200))


def best_instance_pair(targets, preds):
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
            if x is None and y is None:
                assert False
            if x is None or y is None:
                connective_j = 0
            else:

                connective_j = spans_to_jaccard(x.connective.spans,
                                                y.connective.spans)
            score += connective_j
        if score > best:
            best_situation = situation
            best = score
    if not best_situation:
        assert False
    return best_situation


total = 0
metric_1 = FullMetric.create(threshhold=1)
metric_75 = FullMetric.create(threshhold=0.75)
for k, v in a_200.items():
    x = v
    y = b_200[k]
    x = x.causality_instances
    y = y.causality_instances
    if x == {} or y == {}: continue
    total += 1
    # print(x,y)
    # x = x.causality_instances
    x = list(x.values())
    y = list(y.values())
    best_pair = best_instance_pair(x, y)
    for x_instance, y_instance in best_pair:
        if x_instance is None:
            x_instance = CausalityInstance.empty()
        if y_instance is None:
            y_instance = CausalityInstance.empty()
        metric_1.update(x_instance, y_instance)
        metric_75.update(x_instance, y_instance)
        # print(x_instance, y_instance)
print(total)
print(metric_1.report())
print(metric_75.report())

c = copy.deepcopy(a)

# with open('dataset/semeval_600', 'w') as f:
#     data_file_str = jsonpickle.encode(c, indent=2)
#     f.write(data_file_str)

t = 0
for k, v in b.items():
    if k not in c: assert False
    if c[k].is_causal == '' and v.is_causal == 'true':
        t += 1
        c[k].is_causal = 'true'
        c[k].causality_instances = v.causality_instances

print(len(c))
n_causal = 0
n_not_causal = 0
n_instance = 0
n_multiple = 0
length = 0
for k, v in c.items():
    if v.is_causal == 'true': n_causal+=1
    if v.is_causal == 'false': n_not_causal+=1
    if v.is_causal != '': length += len(v.sentence)
    if len(v.causality_instances)>1: n_multiple+=1
    n_instance+= len(v.causality_instances)

print(n_causal)
print(n_not_causal)
print(n_instance)
print(n_multiple)
print(length/(n_causal+n_not_causal))

# with open('dataset/semeval_full', 'w') as f:
#     data_file_str = jsonpickle.encode(c, indent=2)
#     f.write(data_file_str)
# print(t)

    # print(v.is_causal)
