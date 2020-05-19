#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging
import os
from sklearn.model_selection import train_test_split
from common.exceptions import PreprocessingException
from common.utils import remove_dir, create_dir, copy_file, get_pos, get_value


def assign_class(real_value, classes_ranges):
    """
    TODO:  ############################################################################
    :param real_value:
    :param classes_ranges:
    :return:
    """
    val = real_value if isinstance(id, int) else int(real_value)
    for k in classes_ranges.keys():
        if val in range(classes_ranges[k][0], classes_ranges[k][1]+1):
            # print('{} belongs to class "{}"'.format(str(real_value), k))
            return k


def load_unsorted_data(data_path, criteria, file_name_format, classes,
                       classes_ranges):
    """
    TODO:  ############################################################################
    :param data_path:
    :param criteria:
    :param file_name_format:
    :param classes:
    :param classes_ranges:
    :return:
    """
    lab_idx = file_name_format.split('_').index(criteria)
    lab, path = [], []
    files = [os.path.join(data_path, f) for f in os.listdir(data_path)
             if os.path.isfile(os.path.join(data_path, f))
             and f.endswith('.jpg')]
    for f in files:
        path.append(f)
        f_name = f.split('/')[-1]
        lab.append(get_pos(
            assign_class(
                int(f_name.split('_')[lab_idx]), classes_ranges), classes))
    return path, lab


def get_train_test_split(x, y, test_size, random_state=None, shuffle=True,
                         stratify=None):
    """
    TODO:  ############################################################################
    :param x:
    :param y:
    :param test_size:
    :param random_state:
    :param shuffle:
    :param stratify:
    :return:
    """
    return train_test_split(x,
                            y,
                            test_size=test_size,
                            random_state=random_state,
                            shuffle=shuffle,
                            stratify=stratify)


def prepare_data_in_folders(x, y, target_path, classes):
    """
    TODO: ################################################################################
    :param x:
    :param y:
    :param target_path:
    :param classes:
    :return:
    """
    full_path = os.path.abspath(target_path)
    if os.path.exists(full_path):
        # just in case it exists and contains previous executions data
        remove_dir(full_path)
    else:
        create_dir(full_path)
    i = 0
    for f in x:
        fname = f.split('/')[-1]
        cls = get_value(y[i], classes)
        dst = os.path.join(full_path, cls)
        if not os.path.exists(dst):
            create_dir(dst)
        dst = os.path.join(dst, fname)
        copy_file(f, dst)
        i += 1


def save_train_test_sets(inputs=None, targets=None, names=None,
                         dest_path=None, classes=None):
    """
    TODO: ##########################################################################
    :param inputs:
    :param targets:
    :param names:
    :param dest_path:
    :param classes:
    :return:
    """
    if type(inputs) in [tuple, list] and type(targets) in [tuple, list]:
        if len(inputs) == len(targets) == len(names):
            for i in range(len(inputs)):
                path = os.path.join(dest_path, names[i])
                prepare_data_in_folders(inputs[i], targets[i], path, classes)
                i += 1
        else:
            raise PreprocessingException(
                'Inputs size must be equal to targets size')
    else:
        raise PreprocessingException(
            'Please check arguments - inputs: {}, targets: {}, dest_path: {}, '
            'classes: {}'.format(inputs, targets, dest_path, classes))


if __name__ == '__main__':
    pass
