import mne.io


def read_raw_edf(input_fname, exclude, preload, verbose):
    # mne.io.edf.edf.RawEDF seems to not exist, although I can find
    # it here:
    #
    # https://github.com/mne-tools/mne-python/blob/maint/1.6/mne/io/edf/edf.py
    #
    # mne docs say to use mne.io.read_raw_edf instead. see:
    #
    # https://mne.discourse.group/t/how-do-i-process-raw-data-from-edf/7159/2
    return mne.io.read_raw_edf(
        input_fname=input_fname,
        exclude=exclude,
        preload=preload,
        verbose=verbose,
    )
