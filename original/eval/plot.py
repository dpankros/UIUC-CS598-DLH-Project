import os
import matplotlib.pyplot as plt
from eval.reference_stats import get_reference_data
from eval.signals_dict import SignalsDict

def plot_stats(
    our_dict: SignalsDict,
    plots_dir_abs: str,
) -> None:
    """
    Given a signals dictionary and absolute path to a directory in which
    to write plots, process the signals dict, generate plots, and save them
    to the plots directory.
    """
    ref_dict = get_reference_data()
    # we want to get the subset of channels that exist in both our_dict and 
    # ref_dict, then plot performance for those channels
    x_labels = list(our_dict.keyset().intersection(ref_dict.keyset()))
    x_labels_strs = [str(comb) for comb in x_labels]

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
        x_labels_strs,
        ref_dict.means_for_keys("F1", x_labels),
        color="red",
        label="Paper avg. F1"
    )
    plt.plot(
        x_labels_strs,
        ref_dict.means_for_keys("AUROC", x_labels),
        color="black",
        label="Paper avg. AUROC"
    )

    plt.plot(
        x_labels_strs,
        our_dict.means_for_keys("F1", x_labels, float(0)),
        color="cyan",
        label="Our avg. F1"
    )
    
    plt.plot(
        x_labels_strs,
        our_dict.means_for_keys("AUROC", x_labels, float(0)),
        color="green",
        label="Our avg. AUROC",
    )

    plt.legend()

    plt.savefig(os.path.join(plots_dir_abs, "eval.png"))
    
