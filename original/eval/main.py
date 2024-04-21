import os
from config.evals import Evals
from eval.plot import plot_stats
from eval.stats import SignalsDict, StatFile, DEFAULT_INCLUDED_STATS, get_raw_signals_dict


def get_csv_lines_from_files(
    signals_dict: SignalsDict,
    included_stats: list[str] = DEFAULT_INCLUDED_STATS,
) -> list[str]:
    """
    Returns a list of csv lines from a given signals dictionary. 
    :param signals_dict: the collection of signals from which to generate
        CSV lines. You can get one of these by calling 
        get_raw_signals_dict(files_you_care_about)
    :return: a list of csv lines
    """

    csv_lines: list[str] = []
    for signal, stats in signals_dict.dict.items():
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
    signals_dict = get_raw_signals_dict(
        file_list,
    )
    plots_dir = os.path.join(root_dir, "plots")
    plot_stats(signals_dict, plots_dir)
    
    csv_lines = get_csv_lines_from_files(signals_dict)
    print('\n'.join(csv_lines))
