import re
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

from src.data_objects.dataset import (
    CausalityElement,
    CAUSAl_STATE,
    DataPoint,
    CausalityInstance
)
from src.utils.utils import time_to_time_str

DATA = str(Path.home()) / Path('work/brat/data/BECAUSE')
HEARING = DATA / 'CongressionalHearings'
MASC = DATA / 'MASC'


# https://stackoverflow.com/questions/3549075/regex-to-find-all-sentences-of-text
# https://stackoverflow.com/questions/250271/python-regex-how-to-get-positions-and-values-of-matches
# sentence --> word word word word.
# labels --> 1, 2, 3, 1, 1, 3


def read_txt(path):
    with open(path, 'r') as f:
        text = f.read()
    return text


def fill_with_special_character(c):
    def func(m):
        return c * len(m.group())

    return func


fill_with_at = fill_with_special_character('@')
fill_with_space = fill_with_special_character(' ')


def prepare_text_for_sent_split(text):
    for a, b in [
        (r'(?m)^.+:$', fill_with_at),
        (r'\n', r' '),
        (r'Mr\.', 'Mr#'),
        (r'Ms\.', 'Ms#'),
        (r'U.S.', 'U#S#'),
        (r'Rep. Tony', 'Rep# Tony'),
        ('Peter G. Verniero', 'Peter G# Verniero'),
        ('Deborah T. Poritz', 'Deborah T# Poritz'),
        (r'[.](?=\s+?[A-Z]| *?@| *?#| *?\|| *?\'| *?\")', '|'),
        (r'Dr.', 'Dr#')
    ]:
        text = re.sub(a, b, text)
    return text


# typed named tuple
class SentTuple(NamedTuple):
    sent_id: int
    start: int
    end: int
    sent: str


def get_sents(text_for_sent_split):
    pat = re.compile(r"[A-Z][^!?|]*[!?|]", re.M)
    sents_list = []
    sents_iter = pat.finditer(text_for_sent_split)

    # to check ignored text
    start, end = 0, 0

    for i, s in enumerate(sents_iter):

        new_start = s.span()[0]

        # to check ignored text
        # print(text_for_sent_split[end:new_start], end=',')
        # # print(s.group())

        start = s.span()[0]
        end = s.span()[1]
        current_sent = s.group()
        for a, b in [
            ('U#S#', 'U.S.'),
            ('Mr#', 'Mr.'),
            ('Ms#', 'Ms.'),
            ('Dr#', 'Dr.'),
            ('Peter G# Verniero', 'Peter G. Verniero'),
            ('Deborah T# Poritz', 'Deborah T. Poritz'),
            (r'\s+', fill_with_space),
            (r'#{1,}', fill_with_space),
            (r'\|', '.')
        ]:
            current_sent = re.sub(a, b, current_sent)
        assert re.match(r'[A-Z]', current_sent)
        # if '.' in current_sent[:-1]:
        #     print(current_sent)
        sents_list.append(SentTuple(i, start, end, current_sent))
    return sents_list


def check_single_sent(original_sent, current_sent):
    """sent position align with original text"""
    for a, b in [
        ('\n', ' '),
        (r'\s\t+', ' ')
    ]:
        original_sent = re.sub(a, b, original_sent)
    a = original_sent == current_sent
    if not a:
        print(original_sent)
        print(current_sent)
        assert False


def check_sents(original_text, sent_list):
    for s in sent_list:
        check_single_sent(original_text[s.start:s.end], s.sent)


def get_causality_df(path):
    return pd.read_csv(path)


class DfCausalityInstance(NamedTuple):
    ann_id: str
    name: str
    causal: Optional[bool] = None
    continuous: Optional[bool] = None
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None


def search_for_indicator(row, df) -> DfCausalityInstance:
    ann_id = re.search(r'[A-Z][a-z]+:([A-Z]\d{1,10})', row.content).group(1)
    target_rows = df.loc[df['ann_id'] == ann_id, :]
    if not len(target_rows) == 1:
        print(ann_id, target_rows)
        raise ValueError('duplicate indicators')
    assert len(target_rows) == 1
    row = target_rows.iloc[0]
    causal = False if 'NonCausal' in row.content else True
    positions = re.search(r'[A-Z][a-z]+ ([\d ;]+$)', row.content).group(1)
    positions = positions.split(';')
    continuous = False if len(positions) > 1 else True
    start, end = positions[0].split(' ')
    connective_instance = DfCausalityInstance(
        ann_id,
        'indicator',
        causal,
        continuous,
        int(start),
        int(end),
        row.text
    )
    # assert a == connective_instance # for maintaining the same result
    return connective_instance


def get_start_end_continuous_from_positions(positions):
    positions = positions.split(';')
    start, end = positions[0].split(' ')
    continuous = False if len(positions) > 1 else True
    return start, end, continuous


def search_for_cause_or_effect(s, row, core_df, df):
    instances = []
    ann_id = re.search(r'(?<=%s)T\d{1,10}' % s, row.content).group()
    target_rows = df.loc[df['ann_id'] == ann_id, :]
    assert len(target_rows) == 1
    instances = []
    row = target_rows.iloc[0]
    positions = re.search(r'(?<=Argument )[\d ;]+$', row.content).group()
    start, end, continuous = get_start_end_continuous_from_positions(positions)
    instances.append(
        DfCausalityInstance(
            ann_id,
            s[:-1].lower(),
            None,
            continuous,
            int(start),
            int(end),
            row.text)
    )
    corefs = core_df.loc[core_df['content'].str.contains(r':%s[ $]' % ann_id),
             :]
    if len(corefs) > 0:
        for coref in corefs.itertuples():
            id_1 = re.search(
                r'(?<=Arg1:)[A-Z]\d{1,10}(?= )',
                coref.content
            ).group()
            id_2 = re.search(r'(?<=Arg2:)[A-Z]\d{1,10}$', coref.content).group()
            coref_id = id_1 if ann_id == id_2 else id_2
            assert ((ann_id == id_1 and ann_id != id_2)
                    or (ann_id != id_1 and ann_id == id_2))
            tmp_df = df.loc[df['ann_id'] == coref_id, :]
            assert len(tmp_df) == 1
            for _ in tmp_df.itertuples():
                positions = re.search(
                    r'(?<=Argument )[\d ;]+$',
                    row.content
                ).group()
                start, end, coref_continuous = (
                    get_start_end_continuous_from_positions(positions)
                )
                instances.append(DfCausalityInstance(
                    ann_id,
                    s[:-1].lower(),
                    None,
                    coref_continuous,
                    int(start),
                    int(end),
                    row.text)
                )
    # if len(instances) != len(set(instances)):
    #     print(instances)

    # assert res==instances
    return set(instances)


def get_sentence(sents, start, end, triple):
    ret = []
    for s in sents:
        if s.start <= start and end <= s.end:
            ret.append(s)
    if not len(ret) == 1:
        for s in sents:
            print(s)
        print(triple, start, end)
        raise ValueError(
            f'should have found 1 sentence, but found {len(ret)} '
            'sentence(s)'
        )
    return ret[0]


def process_causality(df, sent_list):
    causal = set()
    causality_list = []
    for row in df.itertuples():
        if isinstance(row.ann_id, float):
            assert False
        if (
                row.ann_id.startswith('E')
                and 'Cause' in row.content and 'Effect' in row.content
        ):
            coref_df = df.loc[df['ann_id'].str.contains('R'), :]
            indicator_ans = search_for_indicator(row, df)
            if not indicator_ans.causal:
                continue
            cause_ans = search_for_cause_or_effect('Cause:', row, coref_df, df)
            effect_ans = search_for_cause_or_effect(
                'Effect:',
                row,
                coref_df,
                df
            )
            for c in cause_ans:
                for e in effect_ans:
                    triple = indicator_ans, c, e
                    start = min(
                        (x.start for x in triple if x.start is not None))
                    end = max((x.end for x in triple if x.end is not None))
                    assert indicator_ans.causal
                    sent = get_sentence(sent_list, start, end, triple)
                    if all((x.continuous for x in triple)):
                        causality_list.append((triple, sent))
                    causal.add(sent.sent_id)

    return causal, causality_list


# ['input_ids', 'attention_mask', 'token_type_ids', 'labels']


class ReadyData(NamedTuple):
    uid: str
    sent: str
    indicator: Optional[str]
    cause: Optional[str]
    effect: Optional[str]
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray
    labels: np.ndarray
    spans: Tuple


def is_inside(start, token_span, label_info):
    return (label_info.start <= token_span.start + start
            and token_span.end + start <= label_info.end)


def process(text_file, ann_file, tokenizer, data_causal, data_non_causal):
    original_text = read_txt(text_file)
    text_for_sent_split = prepare_text_for_sent_split(original_text)
    # assert len(original_text) == len(text_for_sent_split)
    sent_list = get_sents(text_for_sent_split)
    # print(len(sent_list))
    # for sent in sent_list:
        # print(sent)
    df = pd.read_csv(
        ann_file,
        sep='\t',
        names=['ann_id', 'content', 'text'],
        header=None
    )
    causal, causality_list = process_causality(df, sent_list)
    # print('clist')
    # print(causality_list)
    # print('causal')
    # print(causal)
    for item in causality_list:
        triple, sent_tuple = item
        # print(triple)
        # print(sent_tuple)
        i, c, e = triple
        sent = sent_tuple.sent
        start = sent_tuple.start
        tokenizer_result = tokenizer(
            sent,
            padding='max_length',
            max_length=256,
            truncation=True
        )
        input_ids = np.array(tokenizer_result.input_ids)
        attention_mask = np.array(tokenizer_result.attention_mask)
        token_type_ids = np.array(tokenizer_result.token_type_ids)
        labels = np.full(256, -100, dtype=int)
        for idx, token in enumerate(tokenizer_result.tokens()):
            if token == '[CLS]':
                continue
            elif token == '[SEP]':
                break
            else:
                token_span = tokenizer_result.token_to_chars(idx)
                compare = token[2:] if token.startswith('#') else token
                if not sent[token_span.start:token_span.end].lower() == compare:
                    print(sent[token_span.start:token_span.end].lower())
                    print(token)
                    assert False
                right_label = 0
                for label, label_info in enumerate([i, c, e], start=1):
                    if is_inside(start, token_span, label_info):
                        if not right_label == 0:
                            # print(d, span[0] + sent_start, span[1] +
                            # sent_start, right_label, j)
                            raise ValueError('Two token labels overlap')
                        right_label = label
                check_dict = {1: i, 2: c, 3: e}

                if right_label != 0 and compare not in check_dict[
                    right_label].text.lower():
                    print(compare)
                    print(check_dict[right_label].text)
                    assert False
                labels[idx] = right_label
        # print(list(zip(tokenizer_result.tokens(), labels)))
        for label_int in [0, 1, 2, 3]:
            if np.count_nonzero(labels == label_int) == 0:
                print(i, c, e)
                print(list(zip(tokenizer_result.tokens(), labels)))
                assert False
        # print(sent_tuple, i, c, e)
        offset = sent_tuple.start
        connective_span = (i.start - offset, i.end - offset)
        cause_span = (c.start - offset, c.end - offset)
        effect_span = (e.start - offset, e.end - offset)
        if i.text != sent[connective_span[0]:connective_span[1]]:
            print(i.text, sent[connective_span[0]:connective_span[1]])
            assert False
        if c.text != sent[cause_span[0]:cause_span[1]]:
            print(c.text, sent[cause_span[0]:cause_span[1]])
            assert False
        if e.text != sent[effect_span[0]:effect_span[1]]:
            print(e.text, sent[effect_span[0]:effect_span[1]])
            assert False
        data_causal.append(ReadyData(
            text_file.name + '_' + i.ann_id + '_' + c.ann_id + '_' + e.ann_id,
            sent, i.text, c.text, e.text, input_ids, attention_mask,
            token_type_ids, labels, (connective_span, cause_span, effect_span)))
    for i, sentence in enumerate(sent_list):
        if i not in causal:
            tokenizer_result = tokenizer(
                sentence.sent,
                padding='max_length',
                max_length=256,
                truncation=True
            )
            input_ids = np.array(tokenizer_result.input_ids)
            attention_mask = np.array(tokenizer_result.attention_mask)
            token_type_ids = np.array(tokenizer_result.token_type_ids)
            labels = np.full(256, -100, dtype=int)
            data_non_causal.append(
                ReadyData(
                    text_file.name + '_' + sentence.sent,
                    sentence.sent,
                    None,
                    None,
                    None,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    labels,
                    tuple()
                )
            )


def split_and_save(df, name):
    train, test = train_test_split(df, test_size=0.2, random_state=10)
    for a, b in [(train, 'train_'), (test, 'test_')]:
        a.reset_index(drop=True, inplace=True)
        a.to_pickle('data_process/dataframe/' + b + name + '.pkl')


def save_causal_non_causal(causal, non_causal, name):
    for a, b in [(causal, 'causal_t_'), (non_causal, 'causal_f_')]:
        split_and_save(a, b + name)


# pandas example
# Pandas(
#     Index=835,
#     uid='Article247_327.txt_That comes to a total of six .',
#     sentence='That comes to a total of six .',
#     connective=None,
#     cause=None,
#     effect=None
# )
def because_df_to_sentence_map(df, start_idx):
    sentence_map = {}
    for row in df.itertuples():
        if row.sentence not in sentence_map:
            start_idx += 1
            if row.connective is None:
                causal = CAUSAl_STATE['not_causal']
            else:
                causal = CAUSAl_STATE['causal']
            data_point = DataPoint(
                uid=row.uid,
                causality_id=start_idx,
                sentence=row.sentence,
                is_causal=causal,
                dataset_origin='because',
            )
            sentence_map[row.sentence] = [data_point, row.uid]
        else:
            data_point = sentence_map[row.sentence][0]
            if not data_point.is_causal:
                # print(data_point)
                # print(row)
                assert row.connective is None
        if not row.connective:
            assert row.cause is None
            assert row.effect is None
            assert len(data_point.causality_instances) == 0
        else:
            connective_span, cause_span, effect_span = row.spans
            connective = CausalityElement(spans=tuple([connective_span]))
            cause = CausalityElement(spans=tuple([cause_span]))
            effect = CausalityElement(spans=tuple([effect_span]))
            key = 'connective_spans_' + str(connective.spans)
            causality_instance = CausalityInstance(
                connective=connective,
                cause=cause,
                effect=effect
            )
            assert key not in data_point.causality_instances
            data_point.causality_instances[key] = causality_instance
    # print(sentence_map.keys())
    return sentence_map


def because_df_to_data_dict(because_df):
    data_dict = {}
    because_sentence_map = because_df_to_sentence_map(
        because_df,
        0
    )
    now = time_to_time_str()
    for key, data_point_and_uid_pair in because_sentence_map.items():
        data_point, uid = data_point_and_uid_pair
        data_point.latest_update_time = now
        data_dict[uid] = data_point
    return data_dict

    # data_file_str = jsonpickle.encode(initial_data_dict, indent=2)
    # with open(initial_data_file_path, 'w') as f:
    #     f.write(data_file_str)
    # print(initial_data_dict)


def create_because_open_data_dict():
    data_causal = []
    data_non_causal = []
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # loop through all files
    for d in DATA.iterdir():
        if d != HEARING and d != MASC: continue
        # print(d)
        for file in d.iterdir():
            # print(file)
            if file.suffix == '.ann':
                ann_file = file
                text_file = file.with_suffix('.txt')
                if Path.is_file(text_file):
                    # print(ann_file, text_file)
                    process(
                        text_file,
                        ann_file,
                        tokenizer,
                        data_causal,
                        data_non_causal
                    )
    # print(len(data_causal))
    # print(len(data_non_causal))
    # print(len(set([x.sent for x in data_causal])))
    # print('info ends')
    # print(data_causal[0])
    # print(data_non_causal[0])
    columns = [
        'uid',
        'sentence',
        'connective',
        'cause',
        'effect',
        'input_ids',
        "attention_mask",
        "token_type_ids",
        'labels',
        'spans'
    ]
    causal = pd.DataFrame(data_causal, columns=columns)
    causal['causal'] = np.array(1)
    non_causal = pd.DataFrame(data_non_causal, columns=columns)
    non_causal['causal'] = np.array(0)
    tdata = pd.concat([causal, non_causal])
    return because_df_to_data_dict(tdata)
