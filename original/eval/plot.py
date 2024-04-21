import os
from math import ceil
import matplotlib.pyplot as plt
from eval.stats import SignalStat, SignalsDict
from eval.reference_stats import get_reference_data

def plot_stats(
    signals_dict: dict[str, dict[str, SignalStat]],
    plots_dir_abs: str,
) -> None:
    """
    Given a signals dictionary and absolute path to a directory in which
    to write plots, process the signals dict, generate plots, and save them
    to the plots directory.
    """
    ref_dict_wrapper = SignalsDict(get_reference_data())
    our_dict_wrapper = SignalsDict(signals_dict)
    x_labels = list(ref_dict_wrapper.dict.keys())

    max_y_axis = max(
        our_dict_wrapper.max(), ref_dict_wrapper.max()
    )
    fig, ax = plt.subplots()
    # ax.bar(x=x_labels, height=max_y_axis)

    plt.xlabel("channel")
    plt.xticks(
        ticks=range(len(x_labels)),
        labels=x_labels,
        rotation=50,
    )

    plt.plot(
        x_labels,
        ref_dict_wrapper.means_for_keys("F1", x_labels),
        color="red",
        label="Paper avg. F1"
    )
    plt.plot(
        x_labels,
        ref_dict_wrapper.means_for_keys("AUROC", x_labels),
        color="black",
        label="Paper avg. AUROC"
    )

    plt.plot(
        x_labels,
        our_dict_wrapper.means_for_keys("F1", x_labels),
        color="cyan",
        label="Our avg. F1"
    )
    
    plt.plot(
        x_labels,
        our_dict_wrapper.means_for_keys("AUROC", x_labels),
        color="green",
        label="Our avg. AUROC",
    )

    plt.legend()

    plt.savefig(os.path.join(plots_dir_abs, "eval.png"))
    
