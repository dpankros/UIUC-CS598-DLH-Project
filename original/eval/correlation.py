from scipy.stats import spearmanr, pearson3
from eval import INDIVIDUAL_CHANS
from eval.signals_dict import SignalsDict
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
    # f = pearson3
    f = spearmanr
    for signal in all_signals:
        for stat in statistics:
            corr, pvalue = f(df[signal], df[stat])

            # Store results in the dictionary
            c = correlation_results.get(signal, {})
            p = p_value_results.get(signal, {})
            c[stat] = corr
            p[stat] = pvalue
            correlation_results[signal] = c
            p_value_results[signal] = p

    return pd.DataFrame(correlation_results), pd.DataFrame(p_value_results)
