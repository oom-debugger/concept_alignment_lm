"Utilities for configuring the training job."
import os
import yaml

def get_yaml_dict(yaml_config):
    config = None
    with open(yaml_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def _update_default(ref_dict, update_dict):
    for k1,v1 in update_dict.items():
        if k1 in ref_dict and isinstance(v1, dict):
            _update_default(ref_dict=ref_dict[k1], update_dict=v1)
        else:
            # we treat lists as same as basic varirables and completely override the default values
            ref_dict[k1] = v1


def get_config(default_filepath, update_filepath):
    config = get_yaml_dict(default_filepath)
    if update_filepath:
        update = get_yaml_dict(update_filepath)
        print(update_filepath)
        _update_default(ref_dict=config, update_dict=update)
    return config
