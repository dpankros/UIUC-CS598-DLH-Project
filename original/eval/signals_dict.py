from math import ceil
from eval.stats import SignalStat
from eval import sorted_chan_str


class NoChannelFoundError(Exception):
    pass

class SignalsDict:
    _vals: dict[str, dict[str, SignalStat]]

    def __init__(self, input_dict: dict[str, dict[str, SignalStat]]):
        d = {}
        for chan_str, stats in input_dict.items():
            chan_alpha = sorted_chan_str(chan_str)
            d[chan_alpha] = stats
        self._vals = d

    def keyset(self) -> set[str]:
        return set(self._vals.keys())

    def items(self) -> list[tuple[str, dict[str, SignalStat]]]:
        return list(self._vals.items())
    
    def max(self) -> int:
        """
        Given a bunch of statistics in the signals_dict, return the maximal
        value across all channels and all statistics.
        """
        max_float: float = 0
        for sig_dict in self._vals.values():
            for sig_stat in sig_dict.values():
                max_float = max(
                    max_float,
                    sig_stat.mean_float(),
                    sig_stat.std_dev_float()
                )
        return ceil(max_float)

    def channels(self) -> list[str]:
        """Return all the channels in this dictionary as strings"""
        return [str(ch) for ch in self._vals.keys()]

    def stats_for_keys(
        self,
        measure: str,
        chans: list[str],
        default: SignalStat | None = None
    ) -> list[SignalStat]:
        """
        Returns a list of SignalStats for
        each signals_dict[chan][measure], where chan is each element in 
        chans. Results will be returned in the same order as specified in
        chans.
        
        If default is None, raises NoChannelFoundError if any channel in chans 
        doesn't exist self. Otherwise, returns a list of means where all 
        missing channels have value default.
        """
        # sort each channel alphabetically, and get a new list in the same
        # order or all the alphabetically-sorted channels
        sorted_chans = [sorted_chan_str(ch) for ch in chans]
        keys = self.channels()
        ret_list: list[SignalStat] = []
        for chan in sorted_chans:
            if chan not in self._vals.keys() and default is None:
                # no default given and the channel isn't in the dict
                raise NoChannelFoundError(
                    f"Dictionary with channels {keys} doesn't contain channel "
                    f"{chan}"
                )
            elif chan not in self._vals.keys() and default is not None:
                # valid default given and channel isn't in the dict
                ret_list.append(default)
            else:
                ret_list.append(self._vals[chan][measure])
        return ret_list
    
    def means_for_keys(
        self,
        measure: str,
        chans: list[str],
        default: float | None = None
    ) -> list[float]:
        """
        Same as stats_for_keys except calls self.dict[chan][measure].mean_float()
        for all chan in chans.

        If default is None, raises NoChannelFoundError if any channel in chans 
        doesn't exist in self. Otherwise, returns a list of means where all 
        missing channels have value default.
        """
        reified_default = SignalStat(str(default), str(default)) if default is not None else None
        signal_stats = self.stats_for_keys(
            measure=measure,
            chans=chans,
            default=reified_default
        )
        return [signal_stat.mean_float() for signal_stat in signal_stats]    

