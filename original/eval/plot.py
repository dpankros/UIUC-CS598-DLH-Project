import os
import matplotlib.pyplot as plt
from eval.signals_dict import SignalsDict, list_by_len_then_alpha

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_stats(
    data: list[SignalsDict]|SignalsDict,
    data_prefixes: list[str],
    plots_dir_abs: str,
    statistics: list[str] =["F1", "AUROC"],
    colors: list[str] = ['red', 'black', 'cyan', 'green'],
    filename: str = 'eval.png',
    size: (float|int, float|int) = (12,4),
    include_legend: bool = True,
) -> None:
    """
    Given a list of signals dictionaries and absolute path to a directory in which
    to write plots, process the signals dict, generate plots, and save them
    to the plots directory.
    """
    if isinstance(data, SignalsDict):
        data = [data]

    if data_prefixes == None or len(data_prefixes) == 0:
        data_prefixes = [""]*len(data)

    x_labels = data[0].keyset()
    for d in data:
        x_labels.intersection(d.keyset())
    x_labels = list(sorted(x_labels, key=list_by_len_then_alpha))

    x_labels_strs = [str(comb) for comb in x_labels]

    fig, _ = plt.subplots()
    fig.set_size_inches(*size)
    # increase bottom spacing to avoid cutting off longer channel combos
    fig.subplots_adjust(bottom=0.4)

    plt.xlabel("channel")
    plt.xticks(
        ticks=range(len(x_labels)),
        labels=x_labels,
        rotation=90,
    )

    for d_ndx, d in enumerate(data):
        for s_ndx, s in enumerate(statistics):
            color_ndx = ((d_ndx * len(statistics)) + s_ndx) % len(colors)
            plt.plot(
                x_labels_strs,
                d.means_for_keys(s, x_labels),
                color=colors[color_ndx],
                label=' '.join([data_prefixes[d_ndx], s])
            )

    if include_legend:
        plt.legend()

    plt.savefig(os.path.join(plots_dir_abs, filename))



def plot_heatmap(
        data: list[SignalsDict]|SignalsDict,
        plots_dir_abs: str,
        title: str | None = None,
        color_map: str | list[str] = 'coolwarm',
        filename: str = 'eval.png',
        size: None | tuple[float|int, float|int] = None,
) -> None:


    df = pd.DataFrame(data)

    # Calculating the correlation matrix
    # corr = df.corr()
    corr = df

    if size is None:
        size = (len(corr.columns), len(corr.index) + 1)
    # Create the matplotlib figure and axis\
    fig, ax = plt.subplots(figsize=size)

    # Choose a colormap
    cmap = color_map if isinstance(color_map, str) else mcolors.LinearSegmentedColormap.from_list("", color_map)

    # Create the heatmap
    cax = ax.matshow(corr, cmap=cmap)

    # Create colorbar
    fig.colorbar(cax)

    # Set ticks
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index))) # this should be rows, not columns

    # Label the ticks
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)

    # Rotate the tick labels for x-axis
    plt.xticks(rotation=45)

    # Setting the x-axis and y-axis limits
    ax.set_xlim(-0.5, len(corr.columns) - 0.5)
    ax.set_ylim(len(corr.index) - 0.5, -0.5)

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(corr):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    # Add title
    if title:
        plt.title(title)

    # Show the plot
    plt.savefig(os.path.join(plots_dir_abs, filename))

