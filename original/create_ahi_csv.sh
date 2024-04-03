#!/usr/bin/env bash

# usage <this file> [path] where path is a path to sleep study data
# redirect stdout to the CSV file you want to create e.g.:
# create_ahi_csv.sh "../../data/Sleep_Data" > AHI.csv

DIR=${1:-"."}
DIR=$(realpath $DIR)

AHI_VALUE="10"

printf 'Study,AHI\n'
for f in $DIR/*.edf; do
  echo "$f" | sed -E "s/^(.*\/)?([0-9]+_[0-9]+).edf$/\"\2\", $AHI_VALUE/g"
done
