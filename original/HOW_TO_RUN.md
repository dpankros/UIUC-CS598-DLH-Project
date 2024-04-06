# How to run this code

(Or How to Hate Yourself in D-Minor)

1. Download data from [physionet.org](https://physionet.org) (or chat)
1. Run `export DATA_ROOT=<path to your data directory>`
    - Make sure this points to the directory that contains your `physionet.org`
    - Dave Uses: `/Volumes/project/data/physionet.org/` on Mac and `/mnt/e/data/`physionet.org/` on PC (wsl)
    - Aaron Uses: `/root/data` on Ubuntu (DigitalOcean VM)
1. Run `make dataload` to do the following in order:
    1. Create the AHI.csv file, if it didn’t already exist
        - If you want to force it to be recreated, run `rm $DATA_ROOT/physionet.org/AHI.csv` before running this `make` command
    1. Run `python3 data/nch/preprocessing.py` to compile each sleep study into an `.npz` file
    1. Run `python3 data/nch/dataloader.py` to aggregate the sleep studies into a single `.npz` file.
        - If you want to force this file to be recreated, run `rm $DATA_ROOT/physionet.org/nch_30x64.npz`
1. Run `python3 main_nch.py` **I hope??**

