import os
import matplotlib.pyplot as plt
import numpy as np

# see the following link for a tutorial on bar charts:
# 
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py


def plot_demographics(save_to: str) -> None:
    def process_vals(
        raw_data: dict[str, tuple[int, int]],
        cols: tuple[str, str],
    ) -> tuple[dict[str, list[int]], int]:
        """
        Given a dictionary of keys mapped to the values for each column, and a tuple
        containing each column, return a 2-tuple with the following elements:

        - A dictionary mapping each column to all of its values, in the same order
        as given in the dictionary
        - The maximal int value found in any element of any tuple in the dictionary
        (useful for plotting on an axis)
        """
        (col1_name, col2_name) = cols
        ret_dict: dict[str, list[int]] = {col1_name: [], col2_name: []}
        max_tup_val = 0
        for _, val_tup in raw_data.items():
            ret_dict[col1_name].append(val_tup[0])
            ret_dict[col2_name].append(val_tup[1])
            max_tup_val = max(max_tup_val, val_tup[0], val_tup[1])
    
        return ret_dict, max_tup_val

    # measurement -> (NCHValue, CHATValue)
    raw_data: dict[str, tuple[int, int]] = {
        "Num. Patients": (3673, 453),
        "Num. Sleep Studies": (3984, 453),
        "Sex=Male": (2068, 219),
        "Sex=Female": (1604, 234),
        "Race=Asian": (93, 8),
        "Race=Black": (738, 252),
        "Race=White": (2433, 161),
        "Race=Other": (409, 32),
        # "Age range (years)" / Average Age": ("[0-30]/8.8", "[5-9] / 6.5")
    }

    label_locs = np.arange(len(raw_data.keys()))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')


    vals, max_y_axis = process_vals(raw_data, ("NCH", "CHAT"))
    for attribute, measurement in vals.items():
        offset = width * multiplier
        rects = ax.bar(
            x=label_locs + offset,
            height=measurement,
            width=width,
            label=attribute,
            angle=0
        )
        ax.bar_label(rects, padding=2)
        multiplier += 1


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Data Demographics')
    ax.set_xticks(label_locs + width, raw_data.keys(), rotation="vertical")
    ax.legend(loc='upper center', ncols=3)
    ax.set_ylim(0, max_y_axis + (max_y_axis/8))

    plt.savefig(save_to)

def plot_signals(save_to: str) -> None:
    def process(
        raw_data: list[tuple[str, int, int]],
        col_names: tuple[str, str]
    ) -> tuple[list[str], dict[str, list[int]], int]:
        (col1_name, col2_name) = col_names
        ret_keys: list[str] = []
        ret_dict: dict[str, list[int]] = {col1_name: [], col2_name: []}
        max_val = 0
        for (key, val1, val2) in raw_data:
            ret_keys.append(key)
            ret_dict[col1_name].append(val1)
            ret_dict[col2_name].append(val2)
            max_val = max(max_val, val1, val2)
        return (ret_keys, ret_dict, max_val)
        
    # list of (measurement_name, NCHValue, CHATValue)
    raw_data: list[str, tuple[int, int]] = [
        ("Oxygen Desaturation", 215280, 65006),
        ("Oximeter Event", 161641, 9864),
        ("EEG arousal", 146052, 0),
        ("Respiratory Events", 0, 0),
        ("Hypopnea", 14522, 15871),
        ("Obstructive Hypopnea", 42179, 0),
        ("Obstructive apnea", 15782, 7075),
        ("Central apnea", 6938, 3656),
        ("Mixed apnea", 2650, 0),
        ("Sleep Stages", 0, 0),
        ("Wake", 665676, 10282),
        ("N1", 128410, 13578),
        ("N2", 1383765, 19985),
        ("N3", 875486, 9981),
        ("REM", 611320, 3283)
    ]
    (keys, vals, max_y_axis) = process(raw_data, ("NCH", "CHAT"))

    # the label locations
    label_locs = np.arange(len(raw_data))
    # the width of the bars
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')


    for attribute, measurement in vals.items():
        offset = width * multiplier
        rects = ax.bar(
            x=label_locs + offset,
            height=measurement,
            width=width,
            label=attribute,
            angle=0
        )
        # ax.bar_label(rects, padding=2)
        multiplier += 1


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Signals Summary')
    ax.set_xticks(label_locs + width, keys, rotation="vertical")
    ax.legend(loc='upper center', ncols=3)
    ax.set_ylim(0, max_y_axis + (max_y_axis/16))

    plt.savefig(save_to)

if __name__ == "__main__":
    base_dir = "visualizations"
    plot_demographics(os.path.join(base_dir, "demographics.png"))
    plot_signals(os.path.join(base_dir, "signals.png"))
