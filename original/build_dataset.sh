#!/bin/bash

DATA_ROOT=${1:-./data}
DATA_ROOT="$(realpath $DATA_ROOT)"

# These need to be checked.  THey're just placeholders
AHI_FILE_PATH="$DATA_ROOT/physionet.org/AHI.csv"
PREPROCESS_PATH="$DATA_ROOT/physionet.org/nch_30x64" # contains a buunch of npz files
TRAINING_DATA_PATH="$DATA_ROOT/physionet.org/nch_30x64.npz" # one combined npz file

export DLHPROJ_DATA_ROOT="$DATA_ROOT/physionet.org"

if [ ! -f "$AHI_FILE_PATH" ]; then
  printf '1) Building AHI.csv to %s\n' "$AHI_FILE_PATH"
   python3 data/nch/create_ahi.py "$DATA_ROOT/physionet.org" "$AHI_FILE_PATH"

  if [ ! -f "$AHI_FILE_PATH" ]; then
    printf 'Failed\n'
    exit 1
  fi
else
  printf '1) AHI.csv file found at %s \xE2\x9C\x94\n'  "$AHI_FILE_PATH"
fi

if [ ! -d  "$PREPROCESS_PATH" ] || [ "$(find "$PREPROCESS_PATH" -mindepth 1 -type f -name "*.npz" -print | wc -c)" == "0" ]; then
  printf '2) Preprocessing into %s\n' "$PREPROCESS_PATH"
   python3 data/nch/preprocessing.py

  if [ ! -d  "$PREPROCESS_PATH" ] || [ "$(find "$PREPROCESS_PATH" -mindepth 1 -type f -name "*.npz" -print | wc -c)" == "0" ]; then
    printf 'Failed\n'
    exit 1
  fi
else
  printf '2) Preprocessed data found in %s \xE2\x9C\x94\n' "$PREPROCESS_PATH"
fi

if [ ! -f "$TRAINING_DATA_PATH" ]; then
  printf '3) Combining dataset. Generating %s\n' "$TRAINING_DATA_PATH"
   python3 data/nch/dataloader.py
else
  printf '3) Combined dataset found at %s \xE2\x9C\x94\n' "$TRAINING_DATA_PATH"
fi

if [ -f "$TRAINING_DATA_PATH" ]; then
  printf 'Training data build complete.  Final training data is at: %s \xE2\x9C\x94\n' "$TRAINING_DATA_PATH"
else
  printf 'No training file was output.  Please check the logs\n'
fi



