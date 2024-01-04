import os
from pathlib import Path
import shutil
import re
from src.data_objects.dataset import (
    DataPoint,
    CAUSAl_STATE,
    CausalityInstance,
    CausalityElement
)
from src.utils.utils import time_to_time_str

HOME = str(Path.home())
PTB = HOME + '/work/brat/data/BECAUSE/PTB'
IDX_BEHIND_COLON = re.compile(r'(?<=:)(T[0-9]{1,4})(?=.|$)')


def file_idx_suffix_to_ptb_path(idx, suffix):
    return PTB + f'/wsj_{idx}.' + suffix


def paste():
    for item in os.listdir(PTB):
        if not item.endswith('.ann'): continue
        item_idx = item[4:8]
        src = (PTB
               + '/tokenized_source'
               + f'/{item[4:6]}'
               + f'/wsj_{item_idx}.txt'
               )
        assert os.path.isfile(src)
        dst = file_idx_suffix_to_ptb_path(item_idx, 'txt')
        shutil.copy(src, dst)


def txt_line_str_to_sentence_str(line_str):
    sentence = re.sub(r'<en=[0-9]{1,3}>', '', line_str)
    sentence = re.sub(r"(?<=[a-z]) (?=[,.'])", '', sentence)
    sentence = re.sub(r"(?<=[A-Z.]) (?=,)", '', sentence)
    sentence = re.sub(r"(?<=[a-zA-Z0-9]) (?=[-])(?!--)", '', sentence)
    sentence = re.sub(r"(?<=[-`]) (?=[a-zA-Z0-9])", '', sentence)
    sentence = re.sub("(?<=\$) (?=[0-9])", '', sentence)
    sentence = re.sub("(?<=[0-9]) (?=%)", '', sentence)
    # b = re.sub(r"(?<=[a-z]) (?=,|\.|')", '', sentence)
    # assert a==b
    sentence = sentence.replace('\n', '')
    return sentence


def line_str_to_instance_list(line_str):
    line_str = line_str.strip('\n')
    return line_str.split('\t')


def about_causality(idx, entity):
    if not idx.startswith('E') or entity.startswith('NonCausal'): return False
    if (
            'Motivation' not in entity
            and 'Purpose' not in entity
            and 'Consequence' not in entity
    ):
        return False
    if 'Cause' not in entity or 'Effect' not in entity:
        return False
    return True


def entity_list_to_ids(entity_list):
    connective = cause = effect = None
    for e in entity_list:
        if e.startswith('Cause'):
            cause = IDX_BEHIND_COLON.search(e)[0]
        elif e.startswith('Effect'):
            effect = IDX_BEHIND_COLON.search(e)[0]
        else:
            connective = IDX_BEHIND_COLON.search(e)[0]
    assert connective is not None
    if cause is None:
        print(entity_list)
    assert cause is not None
    assert effect is not None
    return connective, cause, effect


def entity_idx_to_entity(idx, ann_entities):
    for a_e in ann_entities:
        curr_idx = a_e[0]
        if idx == curr_idx:
            return a_e
    assert False


def entity_to_str_list(entity):
    res = []
    spans = entity[1]
    string = entity[2]
    spans = re.sub(r'[^0-9; ]', '', spans)
    spans = spans.split(';')
    for span in spans:
        start, end = list(map(int, span.split()))
        res.append([string[0:end - start], [start, end]])
        string = string[end - start:]
        string = string.strip(' ')
    return res


def entity_idx_to_str_list(idx, ann_entities):
    entity = entity_idx_to_entity(idx, ann_entities)
    return entity_to_str_list(entity)


def str_list_in_sentence(st_list, sentence):
    pattern = re.compile(r'[^a-zA-Z]')
    for st in st_list:
        st = st[0]
        st = pattern.sub('', st)
        sentence = pattern.sub('', sentence)
        if st not in sentence: return False
    return True


def instance_str_list_to_sentence(
        connective_str_list,
        cause_str_list,
        effect_str_list,
        sentences
):
    res = []
    for sentence in sentences:
        checks = [
            str_list_in_sentence(string_list, sentence) for string_list in (
                connective_str_list,
                cause_str_list,
                effect_str_list
            )
        ]
        if not all(checks):
            continue
        res.append(sentence)
    if len(res) == 0:
        # print(cause_str_list)
        # print(connective_str_list)
        # print(effect_str_list)
        # print(sentences)
        print(
            'sentence not found, should only happen several times, this is'
            ' a single time'
        )
        print()
        return []
    return res[0]


def span_str_list_to_span_index_list(str_list, sentence):
    res = []
    smallest = float('inf')

    for s, boundary in str_list:
        a, b = boundary
        smallest = min(smallest, a)
    offset = max(smallest - len(sentence), 0)
    # print(offset)

    for s, boundary in str_list:
        # s = s.strip('"')
        # s = s.strip("'")
        # s = s.replace('$', '\$')
        a, b = boundary
        a -= offset
        b -= offset
        if a < 0:
            # print(a, b)
            assert False
        while s != sentence[a:b]:
            a -= 1
            b -= 1
            # print(a,b)
            if a == -1:
                # print(s)
                # print(boundary)
                # print(sentence)
                return []
        # print(s)
        # print(sentence)
        # matches = re.finditer(s, sentence)
        # n_find = 0
        # for m in matches:
        #     n_find += 1
        #     res.append((m.start(), m.end()))
        # print(n_find)
        # # for s in str_list:
        # #     print(repr(s))
        # #     print(s[0])
        # # print(repr(sentence))
        # assert n_find == 1

    return res


def check_str_list(str_list, sentence):
    for string, positions in str_list:
        a, b = positions
        if sentence[a:b] != string:
            # print(a,b)
            # print(sentence[a:b])
            # print(string)
            return False
    return True


def update(str_list):
    for st, po in str_list:
        po[0] -= 1
        po[1] -= 1
        if po[0] < 0 or po[1] < 0:
            # print(str_list)
            return False
    return True


def create_causal_element(str_list):
    arr = []
    for a, b in str_list:
        arr.append(tuple(b))
    return CausalityElement(tuple(arr))


def valid_ann_entity_list_to_causality_instance(
        entity_list,
        sentences,
        ann_entities,
):
    connective_idx, cause_idx, effect_idx = entity_list_to_ids(entity_list)
    connective_str_list = (
        entity_idx_to_str_list(connective_idx, ann_entities)
    )
    cause_str_list = entity_idx_to_str_list(cause_idx, ann_entities)
    effect_str_list = entity_idx_to_str_list(effect_idx, ann_entities)
    sentence = instance_str_list_to_sentence(
        connective_str_list,
        cause_str_list,
        effect_str_list,
        sentences
    )
    while not all(check_str_list(x, sentence) for x in [
        connective_str_list,
        cause_str_list,
        effect_str_list
    ]):
        for x in [connective_str_list, cause_str_list, effect_str_list]:
            success = update(x)
            if not success:
                # print(connective_str_list)
                # print(cause_str_list)
                # print(effect_str_list)
                # print(sentence)
                # for i, l in enumerate(sentence):
                #     print(i, l)
                return sentence, '', None
    # for a, b in zip(causes, cause_str_list):
    #     s, e = a
    #     assert sentence[s:e] == b
    for x in [
        connective_str_list,
        cause_str_list,
        effect_str_list
    ]:
        assert check_str_list(x, sentence)
    connective = create_causal_element(connective_str_list)
    cause = create_causal_element(cause_str_list)
    effect = create_causal_element(effect_str_list)

    key = 'connective_spans_' + str(connective.spans)
    instance = CausalityInstance(
        connective=connective,
        cause=cause,
        effect=effect
    )
    return sentence, key, instance


def sentences_ann_entities_to_current_file_data_dict(
        sentences,
        ann_entities
):
    unmatch = 0
    res = {}
    for sent in sentences:
        res[sent] = {}

    for a_e in ann_entities:
        idx, point_entities = a_e[0], a_e[1]
        if len(a_e) != 2 or not about_causality(idx, point_entities): continue
        entity_list = point_entities.split()
        if len(entity_list) != 3: continue
        sentence, key, instance = (
            valid_ann_entity_list_to_causality_instance(
                entity_list,
                sentences,
                ann_entities
            )
        )
        if not key:
            unmatch += 1
            continue
        assert sentence in res
        res[sentence][key] = instance
    return res


def process_item(item):
    item_idx = item[4:8]
    ann_file = file_idx_suffix_to_ptb_path(item_idx, 'ann')
    txt_file = file_idx_suffix_to_ptb_path(item_idx, 'txt')
    with open(txt_file, 'r') as f: txt_lines = f.readlines()
    with open(ann_file, 'r') as f: ann_lines = f.readlines()
    sentences = list(map(txt_line_str_to_sentence_str, txt_lines))
    ann_entities = list(map(line_str_to_instance_list, ann_lines))
    # causality instance:
    # sentence_str, connective_str, cause_str, effect_str
    res = (
        sentences_ann_entities_to_current_file_data_dict(
            sentences,
            ann_entities
        )
    )
    return item[:8], res


def create_ptb():
    res = {}
    idx = 0
    for item in os.listdir(PTB):
        if not item.endswith('.ann'): continue
        prefix, points = process_item(item)
        for sentence, instances in points.items():
            idx += 1
            if len(instances) == 0:
                is_causal = CAUSAl_STATE['not_causal']
            else:
                is_causal = CAUSAl_STATE['causal']
            uid = prefix + '_' + sentence
            point = DataPoint(
                uid=uid,
                causality_id=idx,
                sentence=sentence,
                is_causal=is_causal,
                dataset_origin='because',
                latest_update_time=time_to_time_str(),
                causality_instances=instances
            )
            res[uid] = point

    return res
    # point = DataPoint(
    #     uid = uid,
    #     causality_id = cid,
    #     sentence = sentence,
    #     is_causal= CAUSAl_STATE['causal'],
    #     dataset_origin = 'because',
    #     latest_update_time= time_to_time_str(),
    #     causality_instances={}
    # )

    # class DataPoint:
    #     uid: str
    #     causality_id: int
    #     sentence: str
    #     is_causal: str = CAUSAl_STATE['not_determined']
    #     semeval_comment: str = ''
    #     comment: str = ''
    #     sentence_origin: str = ''
    #     dataset_origin: str = ''
    #     latest_update_time: str = ''
    #     # Dict[connective as str(CausalityElement): instance]
    #     causality_instances: Dict[str, CausalityInstance] = field(
    #         default_factory=dict
    #     )
