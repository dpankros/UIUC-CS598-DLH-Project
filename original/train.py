import keras
import keras.metrics
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.losses import BinaryCrossentropy
from sklearn.utils import shuffle
import numpy.typing as npt

from models.models import get_model

THRESHOLD = 1
FOLD = 5

def lr_schedule(epoch, lr):
    if epoch > 50 and (epoch - 1) % 5 == 0:
        lr *= 0.5
    return lr

def _replace_final_dim(orig_x: npt.NDArray, final_dim_size: int) -> npt.NDArray:
    '''
    Given an NDArray `orig_x`, return a new NDArray of the same shape,
    except with final dimension `final_dim_size`.
    '''
    orig_x_shape = orig_x.shape
    x_transform = np.zeros(
        (*orig_x_shape[:-1], final_dim_size)
    )
    assert x_transform.shape[:-1] == orig_x.shape[:-1]
    return x_transform


def train(config, fold: int | None = None):
    print(f'training with config {config}, fold={fold}')

    data = np.load(config["data_path"], allow_pickle=True)

    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    print(
        f'x={x.shape}, y_apnea={y_apnea.shape}, y_hypopnea={y_hypopnea.shape}'
    )
    y = y_apnea + y_hypopnea
    ########################################################################################
    # Channel selection
    # 
    # x has a shape similar to (NUM_FOLDS, 530, 1920, NUM_CHANNELS).
    # each fold is thus shape (530, 1920, NUM_CHANNELS), but we're trying
    # to only extract config["channels"] out into the last dimension.
    # 
    # The new x_transform ndarray is the same shape as the original x, except
    # that its last dimension is the number of channels we're trying to 
    # extract. Since the channels we want to extract comes from configuration,
    # we can't guarantee we'll get the same number of channels as the size 
    # of the final dimension of x, so we have to create a transformed x with
    # the right size in the final dimension.
    chans = config['channels']
    assert (len(x.shape)) == 4
    x_transform = _replace_final_dim(x, len(chans))
    
    
    print(f'Extracting channels {chans}')
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
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        
        replace = x[i][:, :, chans]
        print(
            f'x[{i}].shape = {x[i].shape}, '
            f'assigning x_transform[{i}] to {replace.shape}'
        )
        
        x_transform[i] = replace  # CHANNEL SELECTION

    ########################################################################################
    # we want to select either `fold` if it's not none, and otherwise `FOLD`
    # after we do that selection, take the min of either that or the size
    # of x_transform's first dimension. we'll end up with a number no more 
    # than the number of available folds in x_transform.
    # folds = range(FOLD) if fold is None else range(fold)
    num_folds = min(
        FOLD if fold is None else fold,
        x_transform.shape[0]
    )
    folds = range(num_folds)
    for fold in folds:
        first = True
        for i in range(5):
            if i != fold:
                if first:
                    x_train = x_transform[i]
                    y_train = y[i]
                    first = False
                else:
                    x_train = np.concatenate((x_train, x_transform[i]))
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
        model.save(config["model_path"] + str(fold))
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
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    else:
        model.compile(optimizer="adam", loss=BinaryCrossentropy(),
                      metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    model.fit(x=x_train, y=y_train, batch_size=512, epochs=config["epochs"], validation_split=0.1,
              callbacks=[early_stopper, lr_scheduler])
    ################################################################################################################
    model.save(config["model_path"] + str(0))
    keras.backend.clear_session()
