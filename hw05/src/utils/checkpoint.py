import pickle

import torch


def load_checkpoint(path, map_location=None):
    allow_omegaconf_configs()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False)


def to_plain_container(value):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        DictConfig = ListConfig = ()
        OmegaConf = None

    if OmegaConf is not None and isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: to_plain_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_container(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain_container(item) for item in value)
    return value


def allow_omegaconf_configs():
    if not hasattr(torch.serialization, "add_safe_globals"):
        return
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata
    except ImportError:
        return
    torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
