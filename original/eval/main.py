import os
from config.evals import Evals
from eval.plot import plot_stats, plot_heatmap
from eval.stats import StatFile, DEFAULT_INCLUDED_STATS
from eval.signals_dict import SignalsDict
from eval.files import get_raw_signals_dict
from eval.reference_stats import get_reference_data
from eval.correlation import rank_correlation

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

if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(), "results")
    evals_cfg = Evals(root_dir)
    file_list = [
        StatFile(base_dir=root_dir, file_name=fname)
        for fname in evals_cfg.result_filenames()
        if "_200-" in fname
    ]
    signals_dict = get_raw_signals_dict(
        file_list,
    )

    plots_dir = os.path.join(root_dir, "plots")
    correlation_results, p_value_results = rank_correlation(signals_dict)

    plot_heatmap(correlation_results, plots_dir, title="Correlation of Signal to Statistic", filename="correlation_hm.png")
    plot_heatmap(p_value_results, plots_dir, title="P-Value of Signal to Statistic", filename="p_value_hm.png")
    plot_stats([get_reference_data(), signals_dict], ["Paper avg.", "Our avg."], plots_dir, colors=["royalblue", "darkblue", "coral", "darkred"])
    plot_stats(signals_dict, None, plots_dir, statistics=['F1', 'AUROC'], filename="all_ablations.png", size=(14, 5), colors=["coral", "darkred"])

    csv_lines = get_csv_lines_from_files(signals_dict)
    print('\n'.join(csv_lines))
