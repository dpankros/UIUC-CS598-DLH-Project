import os
from math import ceil
import matplotlib.pyplot as plt
from results.stats import SignalStat
from results.reference_stats import get_reference_data

def _max_signal_stat(signals_dict: dict[str, dict[str, SignalStat]]) -> int:
    max_float: float = 0
    for sig_dict in signals_dict.values():
        for sig_stat in sig_dict.values():
            max_float = max(
                max_float,
                float(sig_stat.mean),
                float(sig_stat.std_dev)
            )
    return ceil(max_float)

def _vals_for_keys(
    signals_dict: dict[str, dict[str, SignalStat]],
    measure: str,
    chans: list[str]
) -> list[SignalStat]:
    """
    Given a signals dictionary, returns a list of SignalStats for
    each signals_dict[chan][measure], where chan is each element in 
    chans. Results will be returned in the same order as specified in
    chants. Raises an exception if any one or more channels doesn't
    exist in signals_dict, or any one channel doesn't contain measure.
    """
    return [
        signals_dict[chan][measure] for chan in chans
    ]

def plot_stats(
    signals_dict: dict[str, dict[str, SignalStat]],
    plots_dir_abs: str,
) -> None:
    """
    Given a signals dictionary and absolute path to a directory in which
    to write plots, process the signals dict, generate plots, and save them
    to the plots directory.
    """
    ref_data = get_reference_data()
    x_labels = list(ref_data.keys())
    ref_y_f1 = [
        float(signal_stat.mean)
        for signal_stat
        in _vals_for_keys(ref_data, "F1", x_labels)
    ]

    max_y_axis = max(
        _max_signal_stat(ref_data),
        _max_signal_stat(signals_dict)
    )
    fig, ax = plt.subplots()
    ax.bar(x=x_labels, height=max_y_axis)

    plt.xlabel("channel")
    plt.xticks(
        ticks=range(len(x_labels)),
        labels=x_labels,
        rotation=50,
    )

    plt.plot(x_labels, ref_y_f1, color="red", label="Paper F1")


    plt.savefig(os.path.join(plots_dir_abs, "eval.png"))
    # for signal, signal_dict in signals_dict.items():
        # pass
        # print(f"signal={signal}: {signal_dict}")

