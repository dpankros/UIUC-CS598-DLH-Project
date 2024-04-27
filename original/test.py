import os
from datetime import datetime

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from metrics import Result
from data.noise_util import add_noise_to_data
from model_name import get_model_name
from channels import transform_for_channels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


THRESHOLD = 1
FOLD = 5


def test(config: dict[str, str], fold=None):
    data = np.load(config["data_path"], allow_pickle=True)
    ############################################################################
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    x_tmp = None
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)

        channels = x[i][:, :, config["channels"]]
        if x_tmp is None:
            x_tmp = np.zeros((FOLD, *channels.shape))

        x_tmp[i] = channels

    x = x_tmp
    ############################################################################
    result = Result()
    folds = range(FOLD) if fold is None else [fold]

    if config["model_path"]:
        model_path = config["model_path"]
    else:
        model_name = get_model_name(config)
        model_path = os.path.join(config["model_dir"], model_name)
    for fold in folds:
        x_test = x[fold]
        if config["test_noise_snr"]:
            x_test = add_noise_to_data(x_test, config["test_noise_snr"])

        y_test = y[
            fold
        ]  # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)
        model = tf.keras.models.load_model(os.path.join(model_path, str(fold)), compile=False)

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(
            predict > 0.5, 1, 0
        )  # For MultiClass np.argmax(y_score, axis=-1)

        result.add(y_test, y_predict, y_score)

    print(
        '\n----------\n'
        'results:\n'
    )
    result.print()
    # model_name = config["model_name"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")  # Format the date and time as a string "YYYYMMDD-HH:mm"
    results_file = os.path.join('results', f"{model_name}-{timestamp}.txt")
    print(
        f'done, saving to {results_file}\n'
        '----------\n'
    )

    result.save(path=results_file, config=config)

    del data, x_test, y_test, model, predict, y_score, y_predict


def test_age_seperated(config):
    x = []
    y_apnea = []
    y_hypopnea = []
    for i in range(10):
        data = np.load(config["data_path"] + str(i) + ".npz", allow_pickle=True)
        x.append(data['x'])
        y_apnea.append(data['y_apnea'])
        y_hypopnea.append(data['y_hypopnea'])
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
        if config["test_noise_snr"]:
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
