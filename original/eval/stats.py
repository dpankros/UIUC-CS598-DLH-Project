import os
import re
from dataclasses import dataclass
from math import ceil

DEFAULT_INCLUDED_STATS = ["F1", "AUROC"]

@dataclass(frozen=True)
class SignalStat:
    """A set of statistics for a single signal"""
    mean: str
    std_dev: str

    def mean_float(self) -> float:
        return float(self.mean)

    def std_dev_float(self) -> float:
        return float(self.std_dev)

@dataclass(frozen=True)
class SignalsDict:
    dict: dict[str, dict[str, SignalStat]]

    def max(self) -> int:
        """
        Given a bunch of statistics in the signals_dict, return the maximal
        value across all channels and all statistics.
        """
        max_float: float = 0
        for sig_dict in self.dict.values():
            for sig_stat in sig_dict.values():
                max_float = max(
                    max_float,
                    sig_stat.mean_float(),
                    sig_stat.std_dev_float()
                )
        return ceil(max_float)

    def stats_for_keys(
        self,
        measure: str,
        chans: list[str],
        default: SignalStat = SignalStat("0", "0")
    ) -> list[SignalStat]:
        """
        Returns a list of SignalStats for
        each signals_dict[chan][measure], where chan is each element in 
        chans. Results will be returned in the same order as specified in
        chans.
        
        Returns default for a given channel if that channel doesn't exist
        """
        return [
            self.dict[chan][measure]
            if chan in self.dict else default
            for chan in chans
        ]
    
    def means_for_keys(
        self,
        measure: str,
        chans: list[str],
        default: float = 0
    ):
        """
        Same as stats_for_keys except calls self.dict[chan][measure].mean_float()
        for all chan in chans.

        Returns default for a given channel if that channel doesn't exist.
        """
        return [
            self.dict[chan][measure].mean_float()
            if chan in self.dict else default
            for chan in chans
        ]



@dataclass(frozen=True)
class StatFile:
    base_dir: str
    file_name: str

    def signal_name(self):
        filename_parts = self.file_name.split("_")
        _, signal, *_ = filename_parts
        return signal

    def full_path(self):
        return os.path.join(self.base_dir, self.file_name)


def get_raw_signals_dict(
    files_list: list[StatFile],
) -> SignalsDict:
    """
    Returns a dictionary with statistics for each channel, built from each
    of the given files.
    :param files_list: a list of filenames
    :param included_stats: a list of stats that are listed in the files
    :return: the statistics dictionary
    """
    # Dictionary with this structure: {signal -> {stat -> SignalStat}}
    raw_dict: dict[str, dict[str, SignalStat]] = {}

    for stat_file in files_list:
        filename = stat_file.file_name
        filename_parts = filename.split('_')
        _, signal, *_ = filename_parts

        file_stats = stats_for_file(stat_file)
        raw_dict[signal] = file_stats
    
    return SignalsDict(raw_dict)


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
