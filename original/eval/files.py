from eval.stats import SignalStat, StatFile, stats_for_file
from eval.signals_dict import SignalsDict


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
