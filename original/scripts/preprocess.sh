#!/bin/bash

DEFAULT_DATA_ROOT="./data"
DATA_ROOT=${1:-$DEFAULT_DATA_ROOT}
DATA_ROOT="$(realpath $DATA_ROOT)"
echo "Using DATA_ROOT='${DATA_ROOT}'"

# the place where the AHI.csv file will be output
AHI_FILE_PATH="$DATA_ROOT/physionet.org/AHI.csv"
# contains a bunch of npz files
PREPROCESS_PATH="$DATA_ROOT/physionet.org/nch_30x64"
# one combined npz file - the output of the data loader/preprocessor
TRAINING_DATA_PATH="$DATA_ROOT/physionet.org/nch_30x64.npz"

export DLHPROJ_DATA_ROOT="$DATA_ROOT/physionet.org"

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



