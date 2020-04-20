# -*- coding: utf-8 -*-
import yaml
from common import Exceptions as ex


def load_config(path):
    """
    Load configuration file. It is expected to be a YAML file
    :param path: Path to the yaml/yml file

    :return:
    """
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)
    raise ex.ConfigurationException(
        "Configuration file type is unknown: {p}".format(p=path))
