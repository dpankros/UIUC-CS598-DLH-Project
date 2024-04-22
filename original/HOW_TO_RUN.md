# How to run this code

The code in the [notebook](../Project.ipynb) should be runnable as-is, but if you plan to make significant modifications to this codebase, you should do so in this directory, outside the notebook, then copy your changes into the notebook. 

Instructions for running the code herein are as follows:

1. Download data from [physionet.org](https://physionet.org) (or chat)
    - You will need to complete a human subjects training and an application, then be approved to access these data
1. Run `export DATA_ROOT=<path to your data directory>`
    - Make sure this points to the directory that contains your `physionet.org`
    - Dave Uses: `/Volumes/project/data/` on Mac and `/mnt/e/data/` on PC (wsl)
    - Aaron Uses: `/root/data` on Ubuntu (DigitalOcean VM)
    - See below for other variables you can configure
1. Run `make dataload` to do the following in order:
    1. Create the AHI.csv file, if it didn’t already exist
        - If you want to force it to be recreated, run `make purge-ahi` before running this `make` command
    1. Run data preprocessing to compile each sleep study into an `.npz` file
    1. Run dataloading to process and aggregate the sleep studies into a single [`.npz`](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file.
        - If you want to force this file to be recreated, run `make purge-dataload`
1. Run `make train`
1. Run `make eval` if you want to run evaluations. This command does two things:
    - Prints to STDOUT a CSV with statistics describing model performance for all channel combinations for which ablations were run. Each line is structured as follows:
    `$CHANNEL_IDENTIFIER,$F1_MEAN,$F1_STD_DEV,$AUROC_MEAN,$AUROC_STD_DEV`. You can send this output to a file with `make eval > evals.csv` and then load it into the spreadsheet program of your choice.
    - Renders a graph comparing the trained model's performance and the performance claimed by the paper. The graph will be available in [`original/results/plots`](./results/plots)

## Other configuration

Below is a list of environment variables you can set before running `make dataload`

| Environment variable name | Data type | Default | Description |
| -- | -- | -- | -- |
| `DLHPROJ_DATA_ROOT` | `string` | `/mnt/e/data` | The location of the root directory containing data. Should contain the `physionet.org` directory |
| `DLHPROJ_MODEL_PATH` | `string` | `./weights/semscnn_ecgspo2/f` | The location to which pretrained models will be written |
| `DLHPROJ_NUM_EPOCHS` | `int` | `100` | The number of epochs for which to train (NOTE: the training script may stop early if it achieves a specific level of performance prior to this number of epochs) |
| `DLHPROJ_FORCE_RETRAIN` | `bool` | `True` | Whether to re-train on a fold, even if a pre-trained model for a given fold is present |
