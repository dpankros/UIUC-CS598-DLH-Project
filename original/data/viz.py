from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# see the following link for a tutorial on bar charts:
# 
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

def _process_vals(
    raw_data: dict[str, tuple[int, int]],
    cols: tuple[str, str],
) -> tuple[dict[str, list[int]], int]:
    (col1_name, col2_name) = cols
    ret_dict: dict[str, list[int]] = {col1_name: [], col2_name: []}
    max_tup_val = 0
    for _, val_tup in raw_data.items():
        ret_dict[col1_name].append(val_tup[0])
        ret_dict[col2_name].append(val_tup[1])
        max_tup_val = max(max_tup_val, val_tup[0], val_tup[1])
    
    return ret_dict, max_tup_val



def plot_demographics(save_to: str):
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


    vals, max_y_axis = _process_vals(raw_data, ("NCH", "CHAT"))
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
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, max_y_axis + (max_y_axis/8))

    plt.savefig(save_to)

if __name__ == "__main__":
    plot_demographics("data_viz/demographics.png")
