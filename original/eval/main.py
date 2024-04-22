import os
from config.evals import Evals
from eval.plot import plot_stats, plot_heatmap
from eval.stats import StatFile, DEFAULT_INCLUDED_STATS
from eval.signals_dict import SignalsDict
from eval.files import get_raw_signals_dict
from eval.reference_stats import get_reference_data


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
    for signal, stats in signals_dict.items():
        line = f"\"{signal}\""
        for s in included_stats:
            line += f",{stats[s].mean},{stats[s].std_dev}"
        csv_lines.append(line)

    return csv_lines

#TODO: move this somewhere else
from scipy.stats import spearmanr, pearson3
from eval import INDIVIDUAL_CHANS
import pandas as pd


def split_channels_from_list(ch_list: str):
    remaining = ch_list
    channels = []
    updated = True # needs to be True to start

    # this isn't the most efficient overall, but the search domain is small so it works.  If it needs optimization
    # we can do it later. It just looks at signals and checks if the "remaining" starts with a known signal.  It
    # removes it from "remaining" if it does, then tries again until remaining is len(0)
    while len(remaining) > 0 and updated:
        updated = False
        for name_to_search in INDIVIDUAL_CHANS:
            if remaining.startswith(name_to_search):
                channels.append(name_to_search)
                remaining = remaining[len(name_to_search):]
                updated = True

    if len(remaining) > 0:
        raise Exception(f"Invalid channel list.  Contains unknown channels: {remaining}")

    return channels

def dataframe_from_signals(signals_dict: SignalsDict) -> pd.DataFrame:
    all_signals = INDIVIDUAL_CHANS

    data = {}
    for signals, stats_dict in signals_dict.items():
        present_signals = split_channels_from_list(signals)

        # set a 1 in the signal row if the signal is used or 0 otherwise
        for s in all_signals:
            data[s] = data.get(s, []) + [1 if s in present_signals else 0]

        for stat, value in stats_dict.items():
            data[stat] = data.get(stat, []) + [value.mean]
    return pd.DataFrame(data)

def rank_correlation(signals_dict: SignalsDict, statistics=['F1', 'AUROC']) -> dict[str, dict[str, float]]:
    # Initialize a dictionary to store correlation results
    correlation_results = {}
    p_value_results = {}
    # List of your independent variables
    all_signals = INDIVIDUAL_CHANS
    df = dataframe_from_signals(signals_dict)

    # Calculate correlation for each signal
    # f = pearson3
    f = spearmanr
    for signal in all_signals:
        for stat in statistics:
            corr, pvalue = f(df[signal], df[stat])

            # Store results in the dictionary
            c = correlation_results.get(signal, {})
            p = p_value_results.get(signal, {})
            c[stat] = corr
            p[stat] = pvalue
            correlation_results[signal] = c
            p_value_results[signal] = p

    return pd.DataFrame(correlation_results), pd.DataFrame(p_value_results)

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
    correlation_results, p_value_results = rank_correlation(signals_dict)

    plot_heatmap(correlation_results, plots_dir, title="Correlation of Signal to Statistic", filename="correlation_hm.png")
    plot_heatmap(p_value_results, plots_dir, title="P-Value of Signal to Statistic", filename="p_value_hm.png")
    plot_stats([get_reference_data(), signals_dict], ["Paper avg.", "Our avg."], plots_dir)
    plot_stats(signals_dict, None, plots_dir, statistics=['F1', 'AUROC'], filename="all_ablations.png")

    csv_lines = get_csv_lines_from_files(signals_dict)
    print('\n'.join(csv_lines))
