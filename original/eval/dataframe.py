from eval import INDIVIDUAL_CHANS
from eval.signals_dict import SignalsDict
from eval.channels import split_channels_from_list
import pandas as pd


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
