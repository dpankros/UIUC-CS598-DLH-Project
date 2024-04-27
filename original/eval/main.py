import os
from config.evals import Evals
from eval.plot import plot_stats, plot_heatmap
from eval.stats import StatFile, DEFAULT_INCLUDED_STATS
from eval.signals_dict import SignalsDict
from eval.files import get_raw_signals_dict
from eval.reference_stats import get_reference_data
from eval.correlation import rank_correlation
from eval.channels import split_channels_from_list

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


def get_md_lines_from_files(
        signals_dict: SignalsDict,
        included_stats: list[str] = DEFAULT_INCLUDED_STATS,
        include_table_header: bool = True,
) -> list[str]:
    """
    Returns a list of csv lines from a given signals dictionary.
    :param signals_dict: the collection of signals from which to generate
        CSV lines. You can get one of these by calling
        get_raw_signals_dict(files_you_care_about)
    :return: a list of csv lines
    """
    checkmark_cols = ["CO2", "ECG", "EEG", "EOG", "RESP", "SPO2"]
    md_lines: list[str] = []

    if include_table_header:
        header_line = "|"
        second_header_line = "|"
        for signal_col in checkmark_cols:
            header_line += f" {signal_col} |"
            second_header_line += f"---|"

        for s in included_stats:
            header_line += f" Mean {s} | StdDev {s} |"
            second_header_line += f"---|---|"

        md_lines.append(header_line)
        md_lines.append(second_header_line)

    for signal, stats in signals_dict.items():
        signals = split_channels_from_list(signal)
        # line = f"\"{signal}\""
        checkmarks = ""
        for signal_col in checkmark_cols:
            checkmarks += " X |" if signal_col in signals else " |"

        line = f"|{checkmarks}"
        for s in included_stats:
            line += f" {stats[s].mean} | {stats[s].std_dev} |"

        md_lines.append(line)

    return md_lines


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

    plot_heatmap(correlation_results, plots_dir, title="Correlation of Signal to Statistic",
                 filename="correlation_hm.png")
    plot_heatmap(p_value_results, plots_dir, title="P-Value of Signal to Statistic", filename="p_value_hm.png")
    plot_stats([get_reference_data(), signals_dict], ["Paper avg.", "Our avg."], plots_dir,
               colors=["royalblue", "darkblue", "coral", "darkred"])
    plot_stats(signals_dict, None, plots_dir, statistics=['F1', 'AUROC'], filename="all_ablations.png", size=(14, 5),
               colors=["coral", "darkred"])

    csv_lines = get_csv_lines_from_files(signals_dict)
    print('\n'.join(csv_lines))

    # enable to generate markdown for the notebook
    # md_lines = get_md_lines_from_files(signals_dict)
    # print('\n'.join(md_lines))
