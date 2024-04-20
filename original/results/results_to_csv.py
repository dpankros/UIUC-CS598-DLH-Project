import re
import os
from dataclasses import dataclass

from config.evals import Evals

@dataclass
class SignalStat:
    """A set of statistics for a single signal"""
    mean: str
    std_dev: str

@dataclass
class StatFile:
    base_dir: str
    file_name: str

    def signal_name(self):
        filename_parts = self.file_name.split("_")
        _, signal, *_ = filename_parts
        return signal

    def full_path(self):
        return os.path.join(self.base_dir, self.file_name)



def stats_for_file(stat_file: StatFile) -> dict[str, SignalStat]:
    with open(stat_file.full_path(), 'r') as file:
        lines = file.readlines()
        stat_obj: dict[str, SignalStat] = {}
        for line in lines:
            # match statistic lines
            match = re.search(r'^([a-z0-9]+):', line, re.I | re.M)
            if not match:
                # we only want to match on lines like the following:
                # 
                # Accuracy: <some number>
                # Precision: <some other number>
                # ...
                continue

            parts = line.split(' ')
            stat, mean, _, stddev, *_ = parts
            stat = stat[0:-1]

            stat_obj[stat] = SignalStat(mean, stddev)
        return stat_obj

def get_raw_signals_dict(
    files_list: list[StatFile],
    included_stats: list[str] = ["F1", "AUROC"],
) -> dict[str, dict[str, SignalStat]]:
    """
    Returns a dictionary with statistics for each channel, built from each
    of the given files.
    :param files_list: a list of filenames
    :param included_stats: a list of stats that are listed in the files
    :return: the statistics dictionary
    """
    # Dictionary with this structure: {signal -> {stat -> SignalStat}}
    signals_dict: dict[str, dict[str, SignalStat]] = {}

    for stat_file in files_list:
        filename = stat_file.file_name
        filename_parts = filename.split('_')
        _, signal, *_ = filename_parts

        file_stats = stats_for_file(stat_file)
        signals_dict[signal] = file_stats
    
    return signals_dict

def get_csv_lines_from_files(
    files_list: list[StatFile],
    included_stats: list[str]=['F1', 'AUROC']
) -> list[str]:
    """
    Returns a list of csv lines from a list of files including only included_stats
    :param files_list: a list of filenames
    :param included_stats: a list of stats that are listed in the files
    :return: a list of csv lines
    """
    # Dictionary with this structure: {signal -> {stat -> SignalStat}}
    signals_dict = get_raw_signals_dict(
        files_list=files_list,
        included_stats=included_stats
    )

    csv_lines: list[str] = []
    for signal, stats in signals_dict.items():
        line = f"\"{signal}\""
        for s in included_stats:
            line += f",{stats[s].mean},{stats[s].std_dev}"
        csv_lines.append(line)

    return csv_lines

if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(), "results")
    evals_cfg = Evals(root_dir)
    file_list = [
        StatFile(base_dir=root_dir, file_name=fname)
        for fname in evals_cfg.result_filenames()
    ]
    csv_lines = get_csv_lines_from_files(file_list)
    print('\n'.join(csv_lines))
