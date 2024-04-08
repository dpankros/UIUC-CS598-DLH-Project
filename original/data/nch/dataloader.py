import glob
import os
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from scipy.interpolate import splev, splrep
from collate import max_dimensions, pad_lists

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


SIGS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
s_count = len(SIGS)

THRESHOLD = 3
FREQ = 64
EPOCH_DURATION = 30
ECG_SIG = 4

# PATH = "D:\\nch_30x64\\"
# OUT_PATH = "D:\\nch_30x64"

_data_root = os.getenv(
    "DLHPROJ_DATA_ROOT",
    '/mnt/e/data/physionet.org'
)
AHI_PATH = os.path.join(_data_root,"AHI.csv")
OUT_PATH = os.path.join(_data_root,"nch_30x64.npz")
PATH = os.path.join(_data_root, "nch_30x64")


def extract_rri(signal, ir, CHUNK_DURATION):
    tm = np.arange(0, CHUNK_DURATION, step=1 / float(ir))  # TIME METRIC FOR INTERPOLATION

    # print('filtering', signal, FREQ)
    # TODO: Temporarily bypassed until we know how we want to handle this
    # filtered, _, _ = st.filter_signal(signal=signal, ftype="FIR", band="bandpass", order=int(0.3 * FREQ),
    #                                   frequency=[3, 45], sampling_rate=FREQ, )
    filtered, _, _ = st.filter_signal(signal=signal, ftype="FIR", band="bandpass", order=int(0.3 * FREQ),
                                      frequency=[3, 30], sampling_rate=FREQ, )
    (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=FREQ)
    (rpeaks,) = correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=FREQ, tol=0.05)

    if 4 < len(rpeaks) < 200:  # and np.max(signal) < 0.0015 and np.min(signal) > -0.0015:
        rri_tm, rri_signal = rpeaks[1:] / float(FREQ), np.diff(rpeaks) / float(FREQ)
        ampl_tm, ampl_signal = rpeaks / float(FREQ), signal[rpeaks]
        rri_interp_signal = splev(tm, splrep(rri_tm, rri_signal, k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(ampl_tm, ampl_signal, k=3), ext=1)

        return np.clip(rri_interp_signal, 0, 2), np.clip(amp_interp_signal, -0.001, 0.002)
    else:
        return np.zeros((FREQ * EPOCH_DURATION)), np.zeros((FREQ * EPOCH_DURATION))


def load_data(path) -> tuple[list[Any], list[Any], list[Any]]:
    # demo = pd.read_csv("../misc/result.csv") # TODO

    ahi = pd.read_csv(AHI_PATH)
    filename = ahi.PatID.astype(str) + '_' + ahi.Study.astype(str)
    ahi_dict = dict(zip(filename, ahi.AHI))
    root_dir = os.path.expanduser(path)
    file_list = os.listdir(root_dir)
    length = len(file_list)

    # print(f"Using AHI from {AHI_PATH}")
    # print(f"Using npz files from {root_dir}")
    # print(f"Files {file_list}")

    study_event_counts = {}
    apnea_event_counts = {}
    hypopnea_event_counts = {}
    ######################################## Count the respiratory events ###########################################
    for i in range(length):
        # skip directories
        if os.path.isdir(file_list[i]):
            continue

        # print(f"Processing {file_list[i]}")
        try:
            # parts = file_list[i].split("_")
            # parts[0] should be nch

            patient_id = (file_list[i].split("_")[0])
            study_id = (file_list[i].split("_")[1])
            apnea_count = int((file_list[i].split("_")[2]))
            hypopnea_count = int((file_list[i].split("_")[3]).split(".")[0])
        except Exception as e:
            print(f"Filename mismatch. Skipping {file_list[i]} ({e})", file=sys.stderr)
            continue
        filename = f"{patient_id}_{study_id}"
        ahi_value = ahi_dict.get(filename, None)
        if ahi_value is None:
            print(f"Sleep study {filename} is not found in AHI.csv.  Skipping {file_list[i]}")
            # print(ahi_dict)
            continue

        try:
            if ahi_value > THRESHOLD:
                apnea_event_counts[patient_id] = apnea_event_counts.get(patient_id, 0) + apnea_count
                hypopnea_event_counts[patient_id] = hypopnea_event_counts.get(patient_id, 0) + hypopnea_count
                study_event_counts[patient_id] = study_event_counts.get(patient_id, 0) + apnea_count + hypopnea_count
        except Exception as e:
            print(f"File structure problem.  Skipping {file_list[i]} ({e})", file=sys.stderr)
            continue

        # never do this without a damn good reason
        # else:
        #     os.remove(PATH + file_list[i])

    apnea_event_counts = sorted(apnea_event_counts.items(), key=lambda item: item[1])
    hypopnea_event_counts = sorted(hypopnea_event_counts.items(), key=lambda item: item[1])
    study_event_counts = sorted(study_event_counts.items(), key=lambda item: item[1])

    ################################### Fold the data based on number of respiratory events #########################
    folds = []
    for i in range(5):
        folds.append(study_event_counts[i::5])

    # print('FOLDS:', folds)

    x = []
    y_apnea = []
    y_hypopnea = []
    counter = 0
    for idx, fold in enumerate(folds):
        first = True
        aggregated_data = None
        aggregated_label_apnea = None
        aggregated_label_hypopnea = None
        for patient in fold:
            counter += 1
            # print(counter)
            glob_path = os.path.join(PATH, patient[0] + "_*")
            # print("glob path", glob_path)
            for study in glob.glob(glob_path):
                study_data = np.load(study)

                signals = study_data['data']
                labels_apnea = study_data['labels_apnea']
                labels_hypopnea = study_data['labels_hypopnea']

                identifier = study.split(os.path.sep)[-1].split('_')[0] + "_" + study.split(os.path.sep)[-1].split('_')[1]
                # print(identifier)
                # demo_arr = demo[demo['id'] == identifier].drop(columns=['id']).to_numpy().squeeze() # TODO

                y_c = labels_apnea + labels_hypopnea
                neg_samples = np.where(y_c == 0)[0]
                pos_samples = list(np.where(y_c > 0)[0])
                ratio = len(pos_samples) / len(neg_samples)
                neg_survived = []
                for s in range(len(neg_samples)):
                    if random.random() < ratio:
                        neg_survived.append(neg_samples[s])
                samples = neg_survived + pos_samples
                signals = signals[samples, :, :]
                labels_apnea = labels_apnea[samples]
                labels_hypopnea = labels_hypopnea[samples]

                data = np.zeros((signals.shape[0], EPOCH_DURATION * FREQ, s_count + 3))
                for i in range(signals.shape[0]):  # for each epoch
                    # data[i, :len(demo_arr), -1] = demo_arr TODO
                    data[i, :, -2], data[i, :, -3] = extract_rri(signals[i, ECG_SIG, :], FREQ, float(EPOCH_DURATION))
                    for j in range(s_count):  # for each signal
                        data[i, :, j] = resample(signals[i, SIGS[j], :], EPOCH_DURATION * FREQ)

                if first:
                    aggregated_data = data
                    aggregated_label_apnea = labels_apnea
                    aggregated_label_hypopnea = labels_hypopnea
                    first = False
                else:
                    aggregated_data = np.concatenate((aggregated_data, data), axis=0)
                    aggregated_label_apnea = np.concatenate((aggregated_label_apnea, labels_apnea), axis=0)
                    aggregated_label_hypopnea = np.concatenate((aggregated_label_hypopnea, labels_hypopnea), axis=0)

        if aggregated_data is not None:
            x.append(aggregated_data.tolist())
        if aggregated_label_apnea is not None:
            y_apnea.append(aggregated_label_apnea.tolist())
        if aggregated_label_hypopnea is not None:
            y_hypopnea.append(aggregated_label_hypopnea.tolist())

    return x, y_apnea, y_hypopnea

def list_lengths(lst):
    """
    Gets all the individual lengths of a list
    :param lst:
    :return:
    """
    if isinstance(lst, list):
        # For each item in the list, recursively process it if it's a list
        # Otherwise, the item itself is not counted and is represented as None for non-list items
        sublengths = [list_lengths(item) for item in lst]
        if len([*filter(lambda v: v is not None, sublengths)]) == 0:
            return len(lst)
        # Instead of returning None for non-list items, you could choose to omit them or handle differently
        return len(lst), sublengths  # Return the length of the current list and the structure
    # Return None or some indication for non-list items, if needed
    return None

if __name__ == "__main__":
    x, y_apnea, y_hypopnea = load_data(PATH)

    # these output the maximum size for dimension.
    # If we're going to make this a consistent size without truncating,
    # this is the size to make it
    print(f"Padded X.shape:{max_dimensions(x)}")
    x_norm = pad_lists(x, 0)

    print(f"Padded Y_a shape: {max_dimensions(y_apnea)}")
    y_apnea_norm = pad_lists(y_apnea, 0)

    print(f"Padded Y_h.shape:{max_dimensions(y_hypopnea)}")
    y_hypopnea_norm = pad_lists(y_hypopnea, 0)

    print(f"Saving to {OUT_PATH}")
    np.savez_compressed(
        OUT_PATH,
        x=x_norm,
        y_apnea=y_apnea_norm,
        y_hypopnea=y_hypopnea_norm,
    )
