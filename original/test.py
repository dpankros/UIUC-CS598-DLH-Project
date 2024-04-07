import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from metrics import Result
from data.noise_util import add_noise_to_data

from channels import transform_for_channels

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


THRESHOLD = 1
FOLD = 5


def test(config: dict[str, str], fold=None):
    data = np.load(config["data_path"], allow_pickle=True)
    ############################################################################
    x, y_apnea, y_hypopnea = data["x"], data["y_apnea"], data["y_hypopnea"]
    x_transform = transform_for_channels(x=x, channels=config["channels"])
    y = y_apnea + y_hypopnea

    max_fold = min(FOLD, x_transform.shape[0])
    if max_fold < x_transform.shape[0]:
        print(
            f'WARNING: only looking at the first {max_fold} of '
            f'{x_transform.shape[0]} total folds in X'
        )
    
    # for i in range(FOLD):
    for i in range(max_fold):
        x_transform[i], y[i] = shuffle(x_transform[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x_transform[i] = x[i][:, :, config["channels"]]
    ############################################################################
    result = Result()
    folds = range(FOLD) if fold is None else [fold]
    for fold in folds:
        x_test = x_transform[fold]
        # NOTE: this config key is not set in both `main_chat.py` and
        # `main_nch.py`. if it were, the code under this `if` would fail
        # because there is no `add_noise_to_data` function in this repository.
        if config.get("test_noise_snr"):
           x_test = add_noise_to_data(x_test, config["test_noise_snr"])

        y_test = y[
            fold
        ]  # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)

        model = tf.keras.models.load_model(
            config["model_path"] + str(fold), compile=False
        )

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(
            predict > 0.5, 1, 0
        )  # For MultiClass np.argmax(y_score, axis=-1)

        result.add(y_test, y_predict, y_score)

    result.print()
    result.save("./results/" + config["model_name"] + ".txt", config)

    del data, x_test, y_test, model, predict, y_score, y_predict


def test_age_seperated(config):
    x = []
    y_apnea = []
    y_hypopnea = []
    for i in range(10):
        data = np.load(config["data_path"] + str(i) + ".npz", allow_pickle=True)
        x.append(data["x"])
        y_apnea.append(data["y_apnea"])
        y_hypopnea.append(data["y_hypopnea"])
    ############################################################################
    y = np.array(y_apnea) + np.array(y_hypopnea)
    for i in range(10):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, config["channels"]]
    ############################################################################
    result = Result()

    for fold in range(10):
        x_test = x[fold]
        if config.get("test_noise_snr"):
            x_test = add_noise_to_data(x_test, config["test_noise_snr"])

        y_test = y[
            fold
        ]  # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)

        model = tf.keras.models.load_model(config["model_path"] + str(0), compile=False)

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(
            predict > 0.5, 1, 0
        )  # For MultiClass np.argmax(y_score, axis=-1)

        result.add(y_test, y_predict, y_score)

    result.print()
    result.save("./results/" + config["model_name"] + ".txt", config)

    del data, x_test, y_test, model, predict, y_score, y_predict
