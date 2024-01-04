import json
from typing import Tuple
from pathlib import Path
import jsonpickle

from src.data_process_scripts_parts.read import data_path_to_data_dict
from src.data_objects.dataset import (
    CausalityElement,
    CausalityInstance,
    DataPoint,
    CAUSAl_STATE
)
from src.utils.utils import time_to_time_str

SEMEVAL_COMMENT_KEY = 'Comment:'


# ideally use ordered set, however, since set is small, use list

def id_to_semeval_uid_str(i):
    return f'semeval_2010_task_8_{i}'


def valid_semeval_causality_data_point_lines(point_in_lines):
    if (len(point_in_lines) != 0
            and point_in_lines[1].startswith('Cause-Effect')): return True
    return False


def semeval_causality_data_point_first_line_to_id_and_sentence(line):
    _id, sentence = line.split('\t')
    _id = int(_id)
    sentence = sentence.strip('"')
    for p in ['<e1>', '</e1>', '<e2>', '</e2>']:
        sentence = sentence.replace(p, '')
    assert '<' not in sentence
    assert '>' not in sentence
    return _id, sentence


def semeval_data_point_lines_to_initial_data_point(lines):
    assert len(lines) == 3
    assert lines[2].startswith(SEMEVAL_COMMENT_KEY)
    _id, sentence = (
        semeval_causality_data_point_first_line_to_id_and_sentence(lines[0])
    )
    semeval_comment = lines[2][len(SEMEVAL_COMMENT_KEY):]
    return _id, sentence, semeval_comment


def semeval_2010_task_8_to_initial_data(
        semeval_file_path,
        initial_data_file_path
):
    initial_data_dict = {}
    causality_id = 0
    sentences = set()
    with open(semeval_file_path, 'r') as f:
        semeval_file_str = f.read()
    for i, point in enumerate(semeval_file_str.split('\n\n')):
        lines = point.splitlines()
        if not valid_semeval_causality_data_point_lines(lines): continue
        causality_id += 1
        _id, sentence, semeval_comment = (
            semeval_data_point_lines_to_initial_data_point(lines)
        )
        if sentence in sentences:
            print('semeval duplicate sentence:')
            print(sentence)
            continue
        sentences.add(sentence)
        assert _id == i + 1
        data_point = DataPoint(
            uid=id_to_semeval_uid_str(_id),
            causality_id=causality_id,
            sentence=sentence,
            semeval_comment=semeval_comment,
            dataset_origin='semeval_2010_task_8'

        )
        assert data_point.uid not in initial_data_dict
        initial_data_dict[data_point.uid] = data_point
    jsonpickle_str = jsonpickle.encode(initial_data_dict, indent=2)
    with open(initial_data_file_path, 'w') as f:
        f.write(jsonpickle_str)
    print(f'{len(initial_data_dict)} semeval points written')


# merge doccano:
#     read doccano, as list
#     merge by compare sentence, O(n**2)


def span_add_to_causality_element(span: Tuple[int, int], causality_element):
    assert span not in causality_element.spans
    list_ = list(causality_element.spans)
    list_.append(span)
    list_.sort()
    spans = tuple(list_)
    return CausalityElement(spans=spans)


def doccano_dict_point_to_point_sentence_and_causality_instances(
        doccano_dict_point
):
    if isinstance(doccano_dict_point['label'], list):
        label = doccano_dict_point.pop('label')
        new_label = {'entities': []}
        for ann in label:
            new_label['entities'].append({
                'start_offset': ann[0],
                'end_offset': ann[1],
                'label' : ann[2].replace('e_', 'effect_') if ann[
                    2].startswith('e_') else ann[2]
            })
        doccano_dict_point['label'] = new_label

    sentence = doccano_dict_point['data']
    # print(doccano_dict_point)
    entities = doccano_dict_point['label']['entities']
    # entity to CausalityInstance
    idx_to_causality_group = {}
    # extract all connectives first
    for entity in entities:
        label = entity['label']
        if not label.startswith('connective'): continue
        start = entity['start_offset']
        end = entity['end_offset']
        idx = label[-1]
        if idx not in idx_to_causality_group:
            idx_to_causality_group[idx] = {
                'connective': CausalityElement(spans=tuple([(start, end)]))
            }
        else:
            span_add_to_causality_element(
                (start, end),
                idx_to_causality_group[idx]['connective']
            )
    for entity in entities:
        label = entity['label']
        if label.startswith('connective'): continue
        start = entity['start_offset']
        end = entity['end_offset']
        idx = label[-1]
        if idx not in idx_to_causality_group:
            print("corresponding causality group not found")
            print(idx_to_causality_group)
            print(entity)
            print(doccano_dict_point)
            print(sentence)
            assert False
        causality_group = idx_to_causality_group[idx]

        if 'cause' in label:
            key = 'cause'
        elif 'effect' in label:
            key = 'effect'
        else:
            print('wrong label, not cause nor effect')
            assert False
        if key not in causality_group:
            causality_group[key] = (
                CausalityElement(spans=tuple([(start, end)]))
            )
        else:
            span_add_to_causality_element((start, end), causality_group[key])
    # build causality instances
    causality_instances = {}
    for idx, causality_group in idx_to_causality_group.items():
        if 'cause' not in causality_group or 'effect' not in causality_group:
            print("'cause' or 'effect' not in causality_group")
            print(causality_group)
            print(sentence)
            print(doccano_dict_point)
            assert False
        causality_instance = CausalityInstance(
            connective=causality_group['connective'],
            cause=causality_group['cause'],
            effect=causality_group['effect']
        )
        causality_instances[
            'connective_spans_' + str(causality_group['connective'].spans)] = (
            causality_instance
        )
    return sentence, causality_instances


def doccano_file_path_to_doccano_data_list(doccano_file_path, filter=False):
    doccano_data_list = []
    doccano_sentences = set()
    with open(doccano_file_path, 'r') as f:
        doccano_file_str = f.read()
    for doccano_point_str in doccano_file_str.splitlines():
        try:
            doccano_dict_point = json.loads(doccano_point_str)
        except:
            print("json loads error")
            print(doccano_point_str)
            assert False
        if filter and doccano_dict_point['id']<400: continue
        if filter and doccano_dict_point['id']==1614: continue
        if filter and doccano_dict_point['id'] == 1819: continue
        if filter and doccano_dict_point['id'] == 1889: continue
        if filter and doccano_dict_point['id'] == 1990: continue
        if filter and doccano_dict_point['id'] == 1993 : continue
        if doccano_dict_point['data'] in doccano_sentences:
            print('doccano duplicate sentence:')
            print(doccano_dict_point['data'])
        else:
            doccano_sentences.add(doccano_dict_point['data'])
        sentence, causality_instances = (
            doccano_dict_point_to_point_sentence_and_causality_instances(
                doccano_dict_point
            )
        )
        doccano_data_list.append([sentence, causality_instances])
    return doccano_data_list


# thoughts: functions either links or map to


def find_uid_in_data_dict_for_a_sentence_str(sentence, data_dict):
    res = None
    for data_point_uid, data_point in data_dict.items():
        if sentence == data_point.sentence:
            if res is None:
                res = data_point_uid
            else:
                print('duplicate sentence in data_dict')
                print(res, sentence, data_point.uid)
                assert False
    return res


def each_doccano_initial_data_point_links_causality_instances(
        doccano_data_list,
        initial_data_dict
):
    # limited capability, only add causality instances to each data_process point
    for sentence, causality_instances in doccano_data_list:
        data_point_key = find_uid_in_data_dict_for_a_sentence_str(
            sentence,
            initial_data_dict
        )
        assert hasattr(initial_data_dict[data_point_key], 'is_causal')
        assert data_point_key in initial_data_dict
        initial_data_dict[data_point_key].is_causal = CAUSAl_STATE['causal']
        (
            initial_data_dict[data_point_key]
                .latest_update_time
        ) = time_to_time_str()
        (
            initial_data_dict[data_point_key]
                .causality_instances
        ) = causality_instances


def merge_not_causal(
        path,
        initial_data_dict
):
    with open(path, 'r') as f:
        st = f.read()
    not_causal = jsonpickle.decode(st)
    for k, p in not_causal.items():
        if p['is_causal'] == '@':
            idx = id_to_semeval_uid_str(k)
            initial_data_dict[idx] = DataPoint(
                uid=idx,
                causality_id='not_causal_' + k,
                sentence=p['sent'],
                semeval_comment='',
                dataset_origin='semeval_2010_task_8',
                is_causal=CAUSAl_STATE['not_causal'],
                latest_update_time=time_to_time_str()

            )
    total = 0
    for k, p in initial_data_dict.items():
        if p.is_causal == CAUSAl_STATE['not_causal']:
            total += 1
    print('total non causal', total)


def create_semeval_dataset():
    semeval_2010_task_8_to_initial_data(
        'dataset/TRAIN_FILE.TXT',
        'dataset/initial_semeval'
    )
    data_dict = data_path_to_data_dict(
        'dataset/initial_semeval'
    )
    doccano_data_list = doccano_file_path_to_doccano_data_list(
        'dataset/labels.jsonl'
    )
    each_doccano_initial_data_point_links_causality_instances(
        doccano_data_list,
        data_dict
    )
    merge_not_causal('dataset/not_causal.txt', data_dict)
    data_file_str = jsonpickle.encode(data_dict, indent=2)
    now = time_to_time_str()
    with open(f'dataset/{now}_semeval_backup', 'w') as f:
        f.write(data_file_str)
    with open('dataset/semeval', 'w') as f:
        f.write(data_file_str)


def _create_semeval_dataset(jsonl_path, out_path, filter=False):
    data_dict = data_path_to_data_dict(
        'dataset/initial_semeval'
    )
    doccano_data_list = doccano_file_path_to_doccano_data_list(
        jsonl_path,
        filter=filter
    )
    each_doccano_initial_data_point_links_causality_instances(
        doccano_data_list,
        data_dict
    )
    merge_not_causal('dataset/not_causal.txt', data_dict)
    data_file_str = jsonpickle.encode(data_dict, indent=2)
    now = time_to_time_str()
    with open(out_path, 'w') as f:
        f.write(data_file_str)


if __name__ == '__main__':
    _create_semeval_dataset(
        Path.home() / 'work/data/semeval/self-labeled-1-600/admin.jsonl',
        Path.home() / 'work/data/semeval/self-labeled-1-600/data'
    )
    _create_semeval_dataset(
        Path.home() / 'work/data/semeval/self-labeled-400-1000/admin.jsonl',
        Path.home() / 'work/data/semeval/self-labeled-400-1000/data',
        filter=True
    )
