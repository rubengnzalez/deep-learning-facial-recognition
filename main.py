#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

from common.utils import load_config, init_logger
import os
import preprocess as pr

if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    init_logger(cfg['logging'], cfg['logging']['name'])
    x, y = pr.load_unsorted_data(raw_path, criteria, file_name_format, classes, classes_ranges)
    train_x, test_x, train_y, test_y = pr.get_train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    target_path = '/home/ruben/workspace/tfg/deep-learning-facial-recognition/data'
    dest_path = os.path.join(target_path, criteria, 'training')
    pr.prepare_data_in_folders(train_x, train_y, dest_path, classes)
    dest_path = os.path.join(target_path, criteria, 'test')
    pr.prepare_data_in_folders(test_x, test_y, dest_path, classes)