#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

from common.utils import load_config, init_logger
from preprocessor import Preprocessor

if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    logger = init_logger(cfg['logging'], cfg['logging']['name'])
    logger.info(cfg)
    prep = Preprocessor(cfg['logging']['name'])
    prep.classify_data_in_folders(
        cfg['preprocessor']['classifier']['input']['path'],
        cfg['preprocessor']['classifier']['input']['file_name_format'],
        cfg['preprocessor']['classifier']['output']['criteria'],
        cfg['preprocessor']['classifier']['output']['path'],
        cfg['preprocessor']['classifier']['output']['classes'])