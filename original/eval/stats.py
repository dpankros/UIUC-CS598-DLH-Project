import os
import re
from dataclasses import dataclass

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
