# How to run this code

The code in the [notebook](../Project.ipynb) should be runnable as-is, but if you plan to make significant modifications to this codebase, you should do so in this directory, then copy your changes into the notebook. Instructions for running the code in here are as follows:

1. Download data from [physionet.org](https://physionet.org) (or chat)
1. Run `export DATA_ROOT=<path to your data directory>`
    - Make sure this points to the directory that contains your `physionet.org`
    - Dave Uses: `/Volumes/project/data/physionet.org/` on Mac and `/mnt/e/data/physionet.org/` on PC (wsl)
    - Aaron Uses: `/root/data` on Ubuntu (DigitalOcean VM)
    - See below for other things you can configure
1. Run `make dataload` to do the following in order:
    1. Create the AHI.csv file, if it didn’t already exist
        - If you want to force it to be recreated, run `rm $DATA_ROOT/physionet.org/AHI.csv` before running this `make` command
    1. Run `python3 data/nch/preprocessing.py` to compile each sleep study into an `.npz` file
    1. Run `python3 data/nch/dataloader.py` to aggregate the sleep studies into a single `.npz` file.
        - If you want to force this file to be recreated, run `rm $DATA_ROOT/physionet.org/nch_30x64.npz`
1. Run `make train`

## Other configuration

Below is a list of environment variables you can set before running `make dataload`

| Environment variable name | Data type | Default | Description |
| -- | -- | -- | -- |
| `DLHPROJ_DATA_ROOT` | `string` | `/mnt/e/data` | The location of the root directory containing data. Should contain the `physionet.org` directory |
| `DLHPROJ_MODEL_PATH` | `string` | `./weights/semscnn_ecgspo2/f` | The location to which pretrained models will be written |
| `DLHPROJ_NUM_EPOCHS` | `int` | `100` | The number of epochs for which to train (NOTE: the training script may stop early if it achieves a specific level of performance prior to this number of epochs) |
| `DLHPROJ_FORCE_RETRAIN` | `bool` | `True` | Whether to re-train on a fold, even if a pre-trained model for a given fold is present |
