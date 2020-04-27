#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import yaml
import os
import sys
import shutil
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from common import exceptions as ex


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
            'Section: {s} not found in configuration file'.format(s=section))
    raise ex.ConfigurationException(
        'Configuration file type is unknown: {p}'.format(p=path))


def init_logger(cfg, name='log', level=logging.INFO):
    """
    Initialize Logger based on configuration passed as argument
    :param cfg: Configuration dict
    :param name: name of the logger
    :param level: logging level
    :return: Logger instance
    """
    name = name if name else cfg['name']
    f = '%(asctime)s|%(levelname)s|' + name + \
        '|%(process)d|%(thread)d|%(filename)s|%(funcName)s|%(message)s'
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


def create_dir(full_path, logger_name='log'):
    """
    Wrapper of os.makedirs, more human readable
    :param full_path: Path to the folder to be created
    :param logger_name: logger name
    """
    logger = logging.getLogger(logger_name)
    try:
        os.makedirs(full_path, exist_ok=True)
    except OSError as e:
        logger.error('Error while creating directory {path}: {err}'.format(
                path=full_path, err=e.strerror))
        sys.exit(1)


def copy_file(orig, dest, logger_name='log'):
    """
    Copies a file from given origin path into given destination path
    :param orig: Origin path (including file name)
    :param dest: Destination path (including file name)
    :param logger_name: logger name
    """
    logger = logging.getLogger(logger_name)
    try:
        shutil.copy(orig, dest)
    except OSError as e:
        logger.error('Error while copying file from {orig} to {dest}: '
                     '{err}'.format(orig=orig, dest=dest, err=e.strerror))
        sys.exit(1)


def remove_dir(dir_path, logger_name='log'):
    """
    Removes the directory passed by argument
    :param dir_path: path to the directory
    :param logger_name: logger name
    :return:
    """
    logger = logging.getLogger(logger_name)
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        logger.error('Error while removing directory {path}: {err}'.format(
                path=dir_path, err=e.strerror))
        sys.exit(1)
