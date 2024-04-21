INDIVIDUAL_CHANS = set([
    "EOG",
    "EEG",
    "ECG",
    "RESP",
    "SPO2",
    "CO2"
])

class ChannelCombo:
    _chans: list[str]

    def __init__(self, chan_combo_str: str):
        unordered: list[str] = []
        for ch in INDIVIDUAL_CHANS:
            if ch in chan_combo_str:
                unordered.append(ch)
        
        self._chans = sorted(unordered)
    
    def __str__(self):
        return "".join(self._chans)
