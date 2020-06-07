#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging
import os
from sklearn.model_selection import train_test_split
from common.utils import remove_dir, create_dir, copy_file, get_pos, \
    get_value, load_config, init_logger


class PreprocessingError(Exception):
    """
    Exception to be raised if an error occurs during a Preprocessing task
    """
    pass


class Preprocessor:

    def __init__(self, data_path, criteria, file_name_format, classes,
                 classes_ranges, dest_path, **kwargs):
        self.logger = logging.getLogger(kwargs.get('logger_name')
                                        if kwargs.get('logger_name', False)
                                        else 'log')
        self.criteria = criteria
        self.file_name_format = file_name_format
        self.classes = classes
        self.classes_ranges = classes_ranges
        self.dest_path = dest_path
        self.images, self.labels = self.load_unsorted_data(data_path)

    def load_unsorted_data(self, data_path):
        """
        TODO:  ############################################################################
        :param data_path:
        :return:
        """
        lab_idx = self.file_name_format.split('_').index(self.criteria)
        lab, path = [], []
        files = [os.path.join(data_path, f) for f in os.listdir(data_path)
                 if os.path.isfile(os.path.join(data_path, f))
                 and f.endswith('.jpg')]
        for f in files:
            path.append(f)
            f_name = f.split('/')[-1]
            lab.append(get_pos(
                self.assign_class(int(f_name.split('_')[lab_idx])),
                self.classes))
        return path, lab

    def assign_class(self, real_value):
        """
        TODO:  ############################################################################
        :param real_value:
        :return:
        """
        val = real_value if isinstance(id, int) else int(real_value)
        for k in self.classes_ranges.keys():
            if val in range(self.classes_ranges[k][0],
                            self.classes_ranges[k][1]+1):
                # print('{} belongs to class "{}"'.format(str(real_value), k))
                return k

    def get_train_test_split(self, test_size, random_state=None, shuffle=True,
                             stratify=True):
        """
        TODO:  ############################################################################
        :param test_size:
        :param random_state:
        :param shuffle:
        :param stratify:
        :return:
        """
        return train_test_split(self.images,
                                self.labels,
                                test_size=test_size,
                                random_state=random_state,
                                shuffle=shuffle,
                                stratify=self.labels if stratify else None)

    def prepare_data_in_folders(self, x, y, path):
        """
        TODO: ################################################################################
        :param x:
        :param y:
        :param path:
        """
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            # just in case it exists and contains previous executions data
            remove_dir(full_path)
        else:
            create_dir(full_path)
        i = 0
        for f in x:
            fname = f.split('/')[-1]
            cls = get_value(y[i], self.classes)
            dst = os.path.join(full_path, cls)
            if not os.path.exists(dst):
                create_dir(dst)
            dst = os.path.join(dst, fname)
            copy_file(f, dst)
            i += 1

    def save_train_test_sets(self, inputs=None, targets=None, names=None):
        """
        TODO: ##########################################################################
        :param inputs:
        :param targets:
        :param names:
        :return:
        """
        if type(inputs) in [tuple, list] and type(targets) in [tuple, list]:
            if len(inputs) == len(targets) == len(names):
                for i in range(len(inputs)):
                    path = os.path.join(self.dest_path, names[i])
                    self.prepare_data_in_folders(inputs[i], targets[i], path)
                    i += 1
            else:
                raise PreprocessingError(
                    'Inputs size must be equal to targets size')
        else:
            raise PreprocessingError(
                'Please check arguments - inputs: {}, targets: {}, dest_path:'
                ' {}, classes: {}'
                ''.format(inputs, targets, self.dest_path, self.classes))

    def run(self, test_size, random_state, shuffle=True, stratify=True):
        """
        # TODO: ####################################################################
        :param test_size:
        :param random_state:
        :param shuffle:
        :param stratify:
        :return:
        """
        train_x, test_x, train_y, test_y = \
            self.get_train_test_split(test_size=test_size,
                                      random_state=random_state,
                                      shuffle=shuffle,
                                      stratify=stratify)
        self.save_train_test_sets(inputs=(train_x, test_x),
                                  targets=(train_y, test_y),
                                  names=('training', 'test'))


if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    init_logger(cfg['logging'], cfg['logging']['name'])
    prep = Preprocessor(cfg['data']['raw_path'],
                        cfg['preprocess']['criteria'],
                        cfg['data']['file_name_format'],
                        cfg['data']['classes_list'],
                        cfg['preprocess']['classes_ranges'],
                        cfg['preprocess']['dest_path'])
    prep.run(test_size=cfg['preprocess']['test_size'],
             random_state=cfg['preprocess']['random_state'],
             shuffle=cfg['preprocess']['shuffle'],
             stratify=True)

