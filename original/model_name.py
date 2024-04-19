import os

def get_model_name(config: dict) -> str:
    """
    Given a config, return the corresponding model base path
    :param config:
    :return: strin
    """
    model_name = config["model_name"] # model name includes the channels
    epochs = config["epochs"]

    assert model_name
    assert epochs

    return f"{model_name}_{epochs}"
