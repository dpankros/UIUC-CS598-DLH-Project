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
    return a new string with the exact same elements sorted alphabetically.

    Blow up if chan_str has duplicate channel names or garbage therein.
    """
    consumed = 0
    present: list[str] = []
    # just loop through each channel in INDIVIDUAL_CHANS and for each one
    # found in chan_str, put it into the 'present' list. afterward, make sure we
    # exhausted all characters in chan_str (otherwise, there are either 
    # duplicate channels or there's garbage in chan_str). finally, sort
    # the 'present' list, make it into one long string and return it
    for candidate in INDIVIDUAL_CHANS:
        if candidate in chan_str:
            consumed += len(candidate)
            present.append(candidate)
    assert consumed == len(chan_str)
    present_sorted = sorted(present)
    return "".join(present_sorted)

