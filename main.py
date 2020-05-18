#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

from common.utils import load_config, init_logger
import preprocess as pr

if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    init_logger(cfg['logging'], cfg['logging']['name'])
    x, y = pr.load_unsorted_data(cfg['data']['raw_path'],
                                 cfg['preprocess']['criteria'],
                                 cfg['data']['file_name_format'],
                                 cfg['preprocess']['classes_list'],
                                 cfg['preprocess']['classes_ranges'])
    train_x, test_x, train_y, test_y = pr.get_train_test_split(
        x,
        y,
        test_size=cfg['preprocess']['test_size'],
        random_state=cfg['preprocess']['random_state'],
        shuffle=cfg['preprocess']['shuffle'],
        stratify=y)
    pr.prepare_data_in_folders(train_x,
                               train_y,
                               cfg['preprocess']['training_path'],
                               cfg['preprocess']['classes_list'])
    pr.prepare_data_in_folders(test_x,
                               test_y,
                               cfg['preprocess']['test_path'],
                               cfg['preprocess']['classes_list'])