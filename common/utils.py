# -*- coding: utf-8 -*-
import yaml
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from common import Exceptions as ex


def load_config(path, section=None):
    """
    Load configuration file. It is expected to be a YAML file
    :param path: Path to the yaml/yml file
    :param section: section to be retrieved from YAML file
    :return: Dict containing program configuration
    """
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        if not section:
            return cfg
        elif section in cfg.keys():
            return cfg[section]
        raise ex.ConfigurationException(
            "Section: '{s}' not found in configuration file".format(s=section))
    raise ex.ConfigurationException(
        "Configuration file type is unknown: {p}".format(p=path))


def init_logger(cfg, name=None, level=logging.INFO):
    """
    Initialize Logger based on configuration passed as argument
    :param cfg: Configuration dict
    :param name: name of the logger
    :param level: logging level
    :return:
    """
    name = name if name else cfg['name']
    f = "%(asctime)s|%(levelname)s|" + name + \
        "|%(process)d|%(thread)d|%(filename)s|%(funcName)s|%(message)s"
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter(f))
    logger = logging.getLogger(name)
    logger.addHandler(c_handler)
    logger.setLevel(level if level else logging.INFO)

    if cfg.get('file_handler', False):
        f_name = (cfg['file_name']).format(
            date=datetime.strftime(datetime.now(), '%Y%m%d'))
        full_path = os.path.join(cfg['path'], f_name)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
        f_handler = \
            RotatingFileHandler(full_path, maxBytes=2097152, backupCount=5)
        f_handler.setFormatter(logging.Formatter(f))
        logger.addHandler(f_handler)
    return logger


def create_dir(full_path, exists_ok=True):
    """
    Wrapper of os.makedirs, more human readable
    :param full_path: Path to the folder to be created
    :param exists_ok: If the target directory already exists, an OSError is
     raised when variable set as True
    :return:
    """
    return os.makedirs(os.path.dirname(full_path), exist_ok=exists_ok)