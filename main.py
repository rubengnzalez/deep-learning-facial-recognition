#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

from common.utils import load_config, init_logger
from preprocess import prepare_data_in_folders

if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    logger = init_logger(cfg['logging'], cfg['logging']['name'])
    prepare_data_in_folders(
        cfg['preprocess']['prepare']['input']['path'],
        cfg['preprocess']['prepare']['input']['file_name_format'],
        cfg['preprocess']['prepare']['output']['criteria'],
        cfg['preprocess']['prepare']['output']['path'],
        cfg['preprocess']['prepare']['output']['classes'])
