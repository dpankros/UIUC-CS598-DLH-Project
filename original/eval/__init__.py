INDIVIDUAL_CHANS = set([
    "EOG",
    "EEG",
    "ECG",
    "RESP",
    "SPO2",
    "CO2"
])

def sorted_chan_str(chan_str: str) -> str:
    """
    Given a string composed exclusively of elements of INDIVIDUAL_CHANS,
    return a new string with the exact same elements sorted alphabetically
    """
    consumed = 0
    present: list[str] = []
    for candidate in INDIVIDUAL_CHANS:
        if candidate in chan_str:
            consumed += len(candidate)
            present.append(candidate)
    assert consumed == len(chan_str)
    return "".join(sorted(present))

