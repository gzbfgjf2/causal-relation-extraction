from datetime import datetime


def time_to_time_str():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


