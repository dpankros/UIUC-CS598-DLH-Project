import os
import matplotlib.pyplot as plt
from eval.stats import SignalsDict
from eval.reference_stats import get_reference_data

def plot_stats(
    our_dict: SignalsDict,
    plots_dir_abs: str,
) -> None:
    """
    Given a signals dictionary and absolute path to a directory in which
    to write plots, process the signals dict, generate plots, and save them
    to the plots directory.
    """
    ref_dict = SignalsDict(get_reference_data())
    x_labels = list(ref_dict.dict.keys())

    fig, _ = plt.subplots()
    # increase bottom spacing to avoid cutting off longer channel combos
    fig.subplots_adjust(bottom=0.4)

    plt.xlabel("channel")
    plt.xticks(
        ticks=range(len(x_labels)),
        labels=x_labels,
        rotation=50,
    )

    plt.plot(
        x_labels,
        ref_dict.means_for_keys("F1", x_labels),
        color="red",
        label="Paper avg. F1"
    )
    plt.plot(
        x_labels,
        ref_dict.means_for_keys("AUROC", x_labels),
        color="black",
        label="Paper avg. AUROC"
    )

    plt.plot(
        x_labels,
        our_dict.means_for_keys("F1", x_labels),
        color="cyan",
        label="Our avg. F1"
    )
    
    plt.plot(
        x_labels,
        our_dict.means_for_keys("AUROC", x_labels),
        color="green",
        label="Our avg. AUROC",
    )

    plt.legend()

    plt.savefig(os.path.join(plots_dir_abs, "eval.png"))
    
