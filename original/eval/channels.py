from eval import INDIVIDUAL_CHANS

def split_channels_from_list(ch_list: str):
    remaining = ch_list
    channels = []
    updated = True  # needs to be True to start

    # this isn't the most efficient overall, but the search domain is small so it works.  If it needs optimization
    # we can do it later. It just looks at signals and checks if the "remaining" starts with a known signal.  It
    # removes it from "remaining" if it does, then tries again until remaining is len(0)
    while len(remaining) > 0 and updated:
        updated = False
        for name_to_search in INDIVIDUAL_CHANS:
            if remaining.startswith(name_to_search):
                channels.append(name_to_search)
                remaining = remaining[len(name_to_search):]
                updated = True

    if len(remaining) > 0:
        raise Exception(f"Invalid channel list.  Contains unknown channels: {remaining}")

    return channels
