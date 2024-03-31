def add_noise_to_data[T](a: T) -> T:
    """
    Stand-in function for adding noise to data.

    This function is used in the codebase, hidden behind a
    config flag, and not defined anywhere in the original research
    repository, so we have defined it here as the identity function,
    just so things run properly.
    """
    print("WARNING: 'test_noise_snr' config option is a no-op")
    return a
