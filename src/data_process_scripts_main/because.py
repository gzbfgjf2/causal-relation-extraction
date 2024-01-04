from src.data_process_scripts_main.because_open import create_because_open_data_dict
from src.data_process_scripts_main.because_ptb import create_ptb
import jsonpickle
from src.utils.utils import time_to_time_str
def create_because(
    bopen=False,
    ptb=False
):
    data_dict = {}
    if bopen: data_dict.update(create_because_open_data_dict())
    if ptb: data_dict.update(create_ptb())
    if not data_dict: return
    data_file_str = jsonpickle.encode(data_dict, indent=2)
    now = time_to_time_str()
    with open(f'dataset/{now}_because_backup', 'w') as f:
        f.write(data_file_str)
    with open('dataset/because', 'w') as f:
        f.write(data_file_str)


