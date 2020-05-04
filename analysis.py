#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging
import skimage.io as skd
import os
import numpy as np
import matplotlib.pyplot as plt
from common.utils import init_logger, load_config


def load_data(data_path):
    """
    # TODO: #######################################################################
    :param data_path:
    :return:
    """
    dirs = [x for x in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, x))]
    lab, im = [], []
    for d in dirs:
        label_dir = os.path.join(data_path, d)
        files = [os.path.join(label_dir, f)
                 for f in os.listdir(label_dir)
                 if os.path.isfile(os.path.join(label_dir, f))
                 and f.endswith('.jpg')]
        for f in files:
            im.append(skd.imread(f))
            lab.append(int(d))
    return im, lab


def data_summary(np_data, flags=True, logger_name='log', use_logger=True):
    """

    :param np_data:
    :param flags:
    :param logger_name:
    :param use_logger:
    :return:
    """
    summary = \
        'Np_Dim: {0}, Np_Bytes: {1}, Np_ItemSize: {2}, Shape: {3}'.format(
            np_data.ndim, np_data.nbytes, np_data.itemsize, np_data.shape)
    flgs = 'Data Flags\n' + str(np_data.flags)
    if use_logger:
        logger = logging.getLogger(logger_name)
        logger.info(summary)
        if flags:
            logger.info(flgs)
    else:
        print(summary)
        if flags:
            print(flgs)


if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    logger = init_logger(cfg['logging'], cfg['logging']['name'])
    images, labels = load_data(cfg['analysis']['input']['path'])
    images = np.array(images)
    data_summary(images)
    plt.hist(labels, len(set(labels)))
    plt.show()
