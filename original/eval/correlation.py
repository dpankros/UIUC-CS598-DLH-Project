from scipy.stats import spearmanr
from eval import INDIVIDUAL_CHANS
from eval.signals_dict import SignalsDict, list_by_len_then_alpha
import pandas as pd
from eval.dataframe import dataframe_from_signals


def rank_correlation(signals_dict: SignalsDict, statistics=['F1', 'AUROC']) -> dict[str, dict[str, float]]:
    # Initialize a dictionary to store correlation results
    correlation_results = {}
    p_value_results = {}
    # List of your independent variables
    all_signals = INDIVIDUAL_CHANS
    df = dataframe_from_signals(signals_dict)

    # Calculate correlation for each signal
    for signal in sorted(all_signals, key=list_by_len_then_alpha):
        for stat in statistics:
            corr, pvalue = spearmanr(df[signal], df[stat])

            # Store results in the dictionary
            c = correlation_results.get(signal, {})
            p = p_value_results.get(signal, {})
            c[stat] = corr
            p[stat] = pvalue
            correlation_results[signal] = c
            p_value_results[signal] = p

    return pd.DataFrame(correlation_results), pd.DataFrame(p_value_results)
