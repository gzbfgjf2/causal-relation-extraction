import numpy as np

import pytorch_lightning as pl
from src.data_process_scripts_main.semeval import create_semeval_dataset
from src.experiment_scripts_main.cause_effect_token_tagging import (
    run_cause_effect_token_classification
)
from src.experiment_scripts_main.connective_span_classification import (
    run_connective_span_classification
)
from src.experiment_scripts_main.instance_baseline import (
    run_instance_baseline
)
from src.experiment_scripts_main.cause_effect_boundary_tagging import (
    run_cause_effect_with_boundary_tagging
)

from src.experiment_scripts_main.instance_token_classification import (
    run_instance_real_data,
    run_instance_token_classification
)

from src.experiment_scripts_main.connective_token_classification import (
    run_connective_token_classification
)

from src.data_process_scripts_main.dataset_txt_view import (
    create_txt_view
)
from functools import partial

import argparse


function_map = {
    1: create_semeval_dataset,
    2: partial(
        run_instance_baseline,
        ['dataset/semeval_full'],
        3
    ),
    3: partial(
        run_connective_token_classification,
        ['dataset/semeval_full'],
        3
    ),
    4: partial(
        run_connective_span_classification,
        ['dataset/semeval_full'],
        3
    ),
    5: partial(
        run_cause_effect_token_classification,
        ['dataset/semeval_full'],
        3
    ),
    6: partial(
        run_cause_effect_with_boundary_tagging,
        ['dataset/semeval_full'],
        3
    ),
    7: partial(
        run_instance_token_classification,
        ['dataset/semeval_full'],
        3
    ),
    8: partial(
        create_txt_view,
        ['dataset/semeval_full']
    ),
    9: run_instance_real_data

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("function_idx", type=int, help=str(function_map))
    pl.seed_everything(42)
    np.random.seed(42)
    args = parser.parse_args()
    idx = args.function_idx
    function_map[idx]()

