import gc
import itertools
import os

from test import test
from config.train import ModelPrefix, parse_train_env
from train import train
from model_name import get_model_name

# "EOG LOC-M2",  # 0
# "EOG ROC-M1",  # 1
# "EEG C3-M2",  # 2
# "EEG C4-M1",  # 3
# "ECG EKG2-EKG",  # 4
#
# "RESP PTAF",  # 5
# "RESP AIRFLOW",  # 6
# "RESP THORACIC",  # 7
# "RESP ABDOMINAL",  # 8
# "SPO2",  # 9
# "CAPNO",  # 10

######### ADDED IN THIS STEP #########
# RRI #11
# Ramp #12
# Demo #13


sig_dict = {"EOG": [0, 1],
            "EEG": [2, 3],
            "RESP": [5, 6],
            "SPO2": [9],
            "CO2": [10],
            "ECG": [11, 12],
            "DEMO": [13],
            }

channel_list = [
    ["ECG", "SPO2"]
    # ["EOG"],
    # ["EEG"],
    # ["RESP"],
    # ["SPO2"],
    # ["CO2"],
    # ["ECG"],
    # ["EOG", "EEG", "RESP", "SPO2", "CO2", "ECG"],
]
RUN_ALL_COMBINATIONS = True # True
EXCLUDED_SIGS = ["DEMO"]
# ALLOWED_LENGTHS = [6, 2, 1]
ALLOWED_LENGTHS = [3, 4, 5]


def all_combinations(signal_names: list[str], lengths: (list[int] | None) = None) -> list:
    """
    Return all the combinations of signals from the list of signal_names where the number of signals is listed in lengths
    :param signal_names: a list of all possible signal names
    :param lengths: a list of all possible combination lengths.  For example [1] will return all the single-element
    combinations, [1, 2] would return all single-element combinations and all two-element combinations, and so on.
    :return:
    """
    lengths = lengths if lengths is not None else range(1, len(signal_names) + 1)
    all = []
    for l in lengths:
        for combo in itertools.combinations(signal_names, l):
            all.append([*combo])

    return all

if __name__ == "__main__":
    if RUN_ALL_COMBINATIONS:
        sig_names = [*filter(lambda n: n not in EXCLUDED_SIGS, sig_dict.keys())]
        # override the selected list of channels with all the combinations, except those in EXCLUDED_SIGS
        channel_list = all_combinations(sig_names, ALLOWED_LENGTHS)

    model_env = parse_train_env()
    print(
        f"-----beginning training-----\n"
        f"model_env={model_env}"
        "----------"
    )
    print(
        f"-----run details-----\n"
        f"Total Ablations: {len(channel_list)}\n"
        f"Signal Ablations:"
    )
    for n, ch in enumerate(channel_list):
        print(f"    {n+1} {ch}")
    print("----------\n")

    for n, ch in enumerate(channel_list):
        chs: list[float] = []
        chstr = ""
        for name in ch:
            chstr += name
            chs = chs + sig_dict[name]
        
        train = model_env.to_train_config(
            ModelPrefix.Transformer,
            chstr,
            chs,
        )

        print(
            f"---{n + 1} of {len(channel_list)}----\n"
            f"model_name={model_env.model_path if model_env.model_path else get_model_name(train)}\n"
            f"training channel {chstr}..."
        )
        train(config=train, force_retrain=model_env.force_retrain)
        print(
            f"\ndone training. beginning testing...\n"
            f"----------\n"
        )
        test(train)
        print(
            f'done testing\n'
            '----------'
        )
        gc.collect()
