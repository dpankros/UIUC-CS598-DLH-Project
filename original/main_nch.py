import gc
import itertools
import os

from test import test
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


def all_combinations(signal_names: list, lengths: (list | None) = None) -> list:
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


def getenv_bool(key_name: str, default: bool) -> bool:
    ret_str = os.getenv(key_name, str(default))
    return True if ret_str == "True" else False


if __name__ == "__main__":
    if RUN_ALL_COMBINATIONS:
        sig_names = [*filter(lambda n: n not in EXCLUDED_SIGS, sig_dict.keys())]
        # override the selected list of channels with all the combinations, except those in EXCLUDED_SIGS
        channel_list = all_combinations(sig_names, ALLOWED_LENGTHS)

    data_root = os.getenv(
        "DLHPROJ_DATA_ROOT",
        "/mnt/e/data"
    )
    model_path = os.getenv(
        "DLHPROJ_MODEL_PATH",
        # "./weights/semscnn_ecgspo2/f"
        None
    )
    model_dir = os.getenv(
        "DLHPROJ_MODEL_DIR",
        "./weights"
    )

    n_epochs = int(os.getenv(
        "DLHPROJ_NUM_EPOCHS",
        "100"
    ))
    force_retrain = getenv_bool(
        key_name="DLHPROJ_FORCE_RETRAIN",
        default=False,
    )
    print(
        f"-----beginning training-----\n"
        f"data_root={data_root}\n"
        f"model_path={model_path}\n"
        f"model_dir={model_dir}\n"
        f"num_epochs={n_epochs}\n"
        f"force_retrain={force_retrain}\n"
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
        chs = []
        chstr = ""
        for name in ch:
            chstr += name
            chs = chs + sig_dict[name]
        config = {
            "data_path": f"{data_root}/nch_30x64.npz",
            "model_path": f"{model_path}" if model_path is not None else None,
            "model_dir": f"{model_dir}",
            # "model_name": "sem-mscnn_" + chstr,  # Must be one of: "Transformer", "cnn", "sem-mscnn", "cnn-lstm", "hybrid"
            "model_name": "Transformer_" + chstr,
            # Must be one of: Transformer: "cnn", "sem-mscnn", "cnn-lstm", "hybrid"
            "regression": False,
            "transformer_layers": 5,  # best 5
            "drop_out_rate": 0.25,  # best 0.25
            "num_patches": 30,  # best 30 TBD
            "transformer_units": 32,  # best 32
            "regularization_weight": 0.001,  # best 0.001
            "num_heads": 4,
            "epochs": n_epochs,  # best 200
            "channels": chs,
        }
        print(
            f"---{n + 1} of {len(channel_list)}----\n"
            f"model_name={model_path if model_path else get_model_name(config)}\n"
            f"training channel {chstr}..."
        )
        train(config=config, force_retrain=force_retrain)
        print(
            f"\ndone training. beginning testing...\n"
            f"----------\n"
        )
        test(config)
        print(
            f'done testing\n'
            '----------'
        )
        gc.collect()
