from eval.stats import SignalStat
from eval.signals_dict import SignalsDict

def parse_from_paper(fmt: str) -> dict[str, SignalStat]:
    """
    The paper gives stats in the form:

    $F1_MEAN($F1_STDDEV) $AUROC_MEAN($AUROC_STDDEV)

    Parse this form into a dictionary of the form:

    {
        "F1": SignalStat(mean=$F1_MEAN, std_dev=$F1_STDDEV),
        "AUROC": SignalStat(mean=$AUROC_MEAN, std_dev=$AUROC_STDDEV)
    }
    """

    f1_comb, auroc_comb = fmt.split(" ")
    f1_mean, f1_std_dev = f1_comb[:-1].split("(")
    auroc_mean, auroc_std_dev = auroc_comb[:-1].split("(")
    return {
        "F1": SignalStat(f1_mean, f1_std_dev),
        "AUROC": SignalStat(auroc_mean, auroc_std_dev)
    }

_channel_id_lookup = [
    "EOG", "EEG", "ECG", "RESP", "SPO2", "CO2"
]

_channel_combos = [
    [0], 
    [1],
    [2],
    [3],
    [4],
    [5],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [2, 3],
    [2, 4],
    [2, 5],
    [3, 4],
    [3, 5],
    [4, 5],
    [0, 1, 2, 3, 4, 5]
]


_stat_vals = [
    "75.4(1.5) 79.9(1.1)",
    "72.7(1.3) 77.5(1.0)",
    "73.0(1.2) 80.1(0.6)",
    "76.4(0.8) 85.3(0.7)",
    "78.6(0.9) 87.1(0.7)",
    "67.4(0.2) 75.9(0.8)",
    "77.0(1.4) 81.2(1.0)",
    "76.6(1.0) 83.4(0.8)",
    "79.9(0.9) 87.6(0.6)",
    "79.6(0.8) 87.8(0.6)",
    "76.1(1.5) 83.1(1.2)",
    "75.1(0.8) 81.1(0.8)",
    "79.2(1.0) 86.9(1.3)",
    "78.9(0.6) 87.4(0.6)",
    "72.7(1.0) 79.1(0.8)",
    "77.5(1.1) 85.7(0.9)",
    "80.7(0.4) 88.4(0.4)",
    "75.1(0.9) 81.9(0.6)",
    "78.4(0.9) 87.0(0.8)",
    "75.9(0.5) 84.7(0.7)",
    "79.8(0.8) 87.5(0.6)",
    "82.6(0.5) 90.4(0.4)"
]

def get_chan_combo_str(chan_combo: list[int]) -> str:
    chans_list: list[str] = []
    for chan_id in chan_combo:
        chan_name = _channel_id_lookup[chan_id]
        chans_list.append(chan_name)
    return "".join(chans_list)

def get_reference_data() -> SignalsDict:
    assert len(_channel_combos) == len(_stat_vals)
    chan_dict: dict[str, dict[str, SignalStat]] = {}
    for idx, chan_combo in enumerate(_channel_combos):
        chan_combo_str = get_chan_combo_str(chan_combo)
        assert chan_combo_str not in chan_dict, (
            f"duplicate channel combination '{chan_combo_str}' found"
        )
        chan_dict[chan_combo_str] = parse_from_paper(_stat_vals[idx])
    
    return SignalsDict(chan_dict)
