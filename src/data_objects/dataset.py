from dataclasses import dataclass, field
from typing import Tuple, Dict

CAUSAl_STATE = {
    'not_determined': '',
    'not_causal': 'false',
    'causal': 'true'
}


@dataclass
class CausalityElement:
    spans: Tuple[Tuple[int, int], ...]

    @classmethod
    def empty(cls):
        # span = tuple([-1, -1])
        return cls(tuple())


@dataclass
class CausalityInstance:
    connective: CausalityElement
    cause: CausalityElement
    effect: CausalityElement

    @classmethod
    def empty(cls):
        element = CausalityElement.empty()
        return cls(connective=element, cause=element, effect=element)


@dataclass
class DataPoint:
    uid: str
    causality_id: int
    sentence: str
    is_causal: str = CAUSAl_STATE['not_determined']
    semeval_comment: str = ''
    comment: str = ''
    sentence_origin: str = ''
    dataset_origin: str = ''
    latest_update_time: str = ''
    # Dict[connective as str(CausalityElement): instance]
    causality_instances: Dict[str, CausalityInstance] = field(
        default_factory=dict
    )
