import gc
import os

from test import test
from train import train

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
    ["ECG", "SPO2"],
]

def getenv_bool(key_name: str, default: bool) -> bool:
    ret_str = os.getenv(key_name, str(default))
    return True if ret_str == "True" else False

if __name__ == "__main__":
    data_root = os.getenv(
        "DLHPROJ_DATA_ROOT",
        "/mnt/e/data"
    )
    model_path = os.getenv(
        "DLHPROJ_MODEL_PATH",
        "./weights/semscnn_ecgspo2/f"
    )
    n_epochs = int(os.getenv(
        "DLHPROJ_NUM_EPOCHS",
        "100"
    ))
    force_retrain = getenv_bool(
        key_name="DLHPROJ_FORCE_RETRAIN",
        default=True,
    )
    print(
        f"-----beginning training-----\n"
        f"data_root={data_root}\n"
        f"model_path={model_path}\n"
        f"num_epochs={n_epochs}\n"
        f"force_retrain={force_retrain}\n"
        "----------"
    )
    for ch in channel_list:
        chs = []
        chstr = ""
        for name in ch:
            chstr += name
            chs = chs + sig_dict[name]
        config = {
            "data_path": f"{data_root}/nch_30x64.npz",
            "model_path": f"{model_path}",
            "model_name": "sem-mscnn_" + chstr,
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
            f"----------\n"
            f"training channel {chstr}..."
        )
        train(config=config, fold=0, force_retrain=force_retrain)
        print(
            f"\ndone training. beginning testing...\n"
            f"----------\n"
        )
        test(config, 0)
        print(
            f'done testing\n'
            '----------'
        )
        gc.collect()
