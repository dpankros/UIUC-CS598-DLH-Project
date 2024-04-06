#!/bin/bash

set -e

DATA_ROOT="$1"
AHI_FILE_PATH="$2"

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
