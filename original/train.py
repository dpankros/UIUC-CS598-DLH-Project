import os
import keras
import keras.metrics
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.losses import BinaryCrossentropy
from sklearn.utils import shuffle
import numpy.typing as npt
from model_name import get_model_name

from models.models import get_model
from channels import transform_for_channels
from folds import concat_all_folds

THRESHOLD = 1
FOLD = 5

def lr_schedule(epoch, lr):
    if epoch > 50 and (epoch - 1) % 5 == 0:
        lr *= 0.5
    return lr




def train(
        config,
        fold: int | None = None,
        force_retrain: bool = False
):
    print(f'training with config {config}, fold={fold}')

    data = np.load(config["data_path"], allow_pickle=True)

    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    print(
        f'x={x.shape}, y_apnea={y_apnea.shape}, y_hypopnea={y_hypopnea.shape}'
    )
    y = y_apnea + y_hypopnea
    ########################################################################################
    # Channel selection

    x_tmp = None
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)

        channels = x[i][:, :, config["channels"]]  # CHANNEL SELECTION
        if x_tmp is None:
            x_tmp = np.zeros((FOLD, *channels.shape))

        x_tmp[i] = channels

    x = x_tmp

    # x_transform here is just the channels from the fold
    # y is min(y_apnea + y_hypopnea, 1)  Basically 1, if there is SDB and 0 otherwise

    ########################################################################################
    #
    # The original code for this is taken from the following link:
    #
    # https://github.com/healthylaife/Pediatric-Apnea-Detection/blob/6dc5ec87ef17810c461d4738dd4f46240816999c/train.py#L39-L48
    #
    # I (Aaron) think that in the inner loop, they're just trying to create
    # one big NDArray with the concatenation of all the folds except for the
    # one on which they're currently on in the outer loop.
    #
    # Then, they train on the concatenated array. In other words, the outer
    # loop behaves similarly to epochs, with a small twist.
    #
    # They used to have the logic to do this inside the outer loop,
    # but I pulled it out.
    #
    # also note, the folds selection (commented below) didn't work because
    # they pass fold=0 into this function, which results in no training
    # whatsoever.
    # folds = range(max_fold)
    # folds = range(FOLD) if fold is None else range(fold)

    if config["model_path"]:
        base_model_path = config["model_path"]
    else:
        model_name = get_model_name(config)
        model_dir = config["model_dir"]
        base_model_path = os.path.join(model_dir, model_name)


    folds = range(FOLD) if fold is None else [fold]
    print(f'iterating over {folds} fold(s)')
    for fold in folds:
        model_path = os.path.join(base_model_path, str(fold))

        if os.path.exists(model_path) and not force_retrain:
            print(
                f'Training fold {fold}: force_retrain==False and '
                f'{model_path} already exists, skipping.'
            )
            continue

        first = True
        x_train = None
        y_train = None
        for i in range(5):

            if i != fold:
                if first:
                    x_train = x[i]
                    y_train = y[i]
                    first = False
                else:
                    x_train = np.concatenate((x_train, x[i]))
                    y_train = np.concatenate((y_train, y[i]))

        model = get_model(config)
        if config["regression"]:
            model.compile(optimizer="adam", loss=BinaryCrossentropy())
            early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        else:
            model.compile(optimizer="adam", loss=BinaryCrossentropy(),
                          metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
            early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        model.fit(x=x_train, y=y_train, batch_size=512, epochs=config["epochs"], validation_split=0.1,
                  callbacks=[early_stopper, lr_scheduler])
        ################################################################################################################
        print(f"saving model for fold {fold} to {model_path}")
        model.save(model_path)
        keras.backend.clear_session()


def train_age_seperated(config):
    data = np.load(config["data_path"], allow_pickle=True)
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    ########################################################################################
    for i in range(10):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)

        x[i] = x[i][:, :, config["channels"]]  # CHANNEL SELECTION

    ########################################################################################
    first = True
    for i in range(10):
        if first:
            x_train = x[i]
            y_train = y[i]
            first = False
        else:
            x_train = np.concatenate((x_train, x[i]))
            y_train = np.concatenate((y_train, y[i]))

    model = get_model(config)
    if config["regression"]:
        model.compile(optimizer="adam", loss=BinaryCrossentropy())
        early_stopper = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

    else:
        model.compile(optimizer="adam", loss=BinaryCrossentropy(),
                      metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        early_stopper = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    lr_scheduler = LearningRateScheduler(lr_schedule)
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=512,
        epochs=config["epochs"],
        validation_split=0.1,
        callbacks=[early_stopper, lr_scheduler]
    )
    ################################################################################################################
    model.save(config["model_path"] + str(0))
    keras.backend.clear_session()
