import sys
import os
import os.path
from .preprocessing import HYPOPNEA_EVENT_DICT, APNEA_EVENT_DICT
import pandas as pd
from datetime.datetime import strptime
import csv


def _num_sleep_hours(
    sleep_study_metadata: pd.DataFrame,
    pat_id: str,
    study_id: str,
) -> int:
    sleep_duration_df = ssl.loc[[
        ssl["STUDY_PAT_ID"] == pat_id and 
        ssl["SLEEP_STUDY_ID"] == study_id
    ]]
    sleep_duration_datetime = strptime(sleep_duration_df[
        "SLEEP_STUDY_DURATION_DATETIME"
    ], "%H:%M:%S")
    return sleep_duration_datetime.hour


def _ahi_for_study(
    sleep_study_metadata: pd.DataFrame,
    sleep_study: pd.DataFrame,
    pat_id: str,
    study_id: str,
    
) -> float:
    '''
    calculate the apnea-hypopnea index (AHI) for a given sleep study.
    params:

    sleep_study_lookup:
        the DataFrame that has at least the following columns in order
        from left to right:
        STUDY_PAT_ID,
        SLEEP_STUDY_ID,
        SLEEP_STUDY_START_DATETIME,
        SLEEP_STUDY_DURATION_DATETIME
    sleep_study:
        the data from the sleep study in which we're interested
    pat_id:
        the ID of the patient on whom the given study was done
    study_id:
        the ID of the study
    
    All apnea and hypopnea events will be counted from the sleep_study
    DataFrame, and then divided by the total sleep duration, which 
    will be gotten from the sleep_study_metadata DataFrame. The result will 
    be returned as a float

    For more on AHI, see the following link:

    https://www.sleepfoundation.org/sleep-apnea/ahi
    '''
    df = sleep_study
    hypopnea_keys = set(HYPOPNEA_EVENT_DICT.keys())
    apnea_keys = set(APNEA_EVENT_DICT.keys())
    hypopnea_events = df.loc[df.description in hypopnea_keys]
    apnea_events = df.loc[df.description in apnea_keys]
    total_num_events = len(hypopnea_events) + len(apnea_events)
    sleep_hours = float(_num_sleep_hours(
        sleep_study_metadata,
        pat_id,
        study_id,
    ))
    return float(total_num_events) / sleep_hours


def _parse_ss_tsv_filename(filename: str) -> tuple[str, str]:
    '''
    given a sleep study filename like `10048_24622.tsv`, that represents
    <patient_id>_<sleep_study_id>.tsv, return a 2-tuple containing
    the patient ID in element 1 and sleep study ID in element 2
    '''
    if not filename.endswith(".tsv"):
        raise FileNotFoundError(
            f"expected {filename} to end with .tsv but it didn't"
        )
    underscore_spl = filename[:-4].split("_")
    if len(underscore_spl) != 2:
        raise FileNotFoundError(f'malformed filename {filename}')
    return (pat_id, study_id)


def _write_tsv(out_filename: str, data: list[tuple[str, str, float]]):
    with open(out_filename, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        # in ./preprocessing.py, we need to have at least 'Study'
        # and 'AHI'. Since they chose PascalCase, I extended that usage
        # to patient ID.
        writer.writerow(("PatID", "Study", "AHI"))
        for row in data:
            writer.writerow(row)


def calculate_ahi(
    sleep_study_metadata_file: str,
    sleep_study_root: str,
    out_file: str,
) -> None:
    metadata_df = pd.read_csv(sleep_study_metadata_file)

    tsv_files = [
        f for f in os.listdir(sleep_study_root)
        if f.endswith(".tsv")
    ]

    print(
        f"creating AHI from {len(tsv_files)} in {sleep_study_root}, "
        "outputting to {out_file}"
    )

    # each tuple is (patient_id, study_id, AHI)
    results: list[tuple[str, str, float]] = []
    for tsv_file in tsv_files:
        filename = os.path.join(sleep_study_root, tsv_file)
        pat_id, study_id = _parse_ss_tsv_filename(filename)
        sleep_study_df = pd.read_csv(filename)
        ahi = _ahi_for_study(
            metadata_df,
            sleep_study_df,
            pat_id,
            study_id,
        )
        results.append((pat_id, study_id, ahi))
    _write_tsv(out_file, results)


if __name__ == "__main__":
    # example tsv file:
    # onset duration description
    # 29766.7421875	11.0546875	Obstructive Hypopnea
    def usage():
        print(
            "Usage: python create_ahi.py <sleep study metadata file> "
            "<nch sleep study root> <output file>"
        )
        sys.exit(1)
    if len(sys.argv) != 5:
        usage()
        sys.exit(1)
    calculate_ahi(
        sleep_study_metadata_file=sys.argv[1],
        sleep_study_root=sys.argv[2],
        out_file=sys.argv[3],
    )
