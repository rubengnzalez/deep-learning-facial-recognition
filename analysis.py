#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging
import math
import random
from datetime import datetime
import skimage.io as skd
import os
import numpy as np
import matplotlib.pyplot as plt
from common.utils import init_logger, load_config, create_dir


def load_data(data_path, logger_name='log'):
    """
    # TODO: #######################################################################
    :param data_path:
    :param logger_name:
    :return:
    """
    log = logging.getLogger(logger_name)
    log.info('Analysis - Loading data from path [{}]'.format(data_path))
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
    log.info('Analysis - Loaded {} images and {} different classes'.format(
        len(im), len(set(lab))))
    return im, lab


def data_summary(np_data, flags=True, logger_name='log'):
    """
    # TODO: #########################################################################
    :param np_data:
    :param flags:
    :param logger_name:
    """
    log = logging.getLogger(logger_name)
    summary = 'Analysis - Data Summary [Np_Dim: {0}, Np_Bytes: {1}, ' \
              'Np_ItemSize: {2}, Shape: {3}]'.format(np_data.ndim,
                                                     np_data.nbytes,
                                                     np_data.itemsize,
                                                     np_data.shape)
    log.info(summary)
    if flags:
        log.info('Analysis - Data Flags [{}]'.format(
            str(np_data.flags).replace('\n', '')))


def save_figure(fig, path, name, append_date=True, logger_name='log'):
    """
    # TODO: ##########################################################################
    :param fig:
    :param path:
    :param name:
    :param append_date:
    :param logger_name:
    """
    log = logging.getLogger(logger_name)
    if not os.path.exists(path):
        log.info('Analysis - Path [{}] does not exist. Creating...'
                 ''.format(path))
        create_dir(path)
    if append_date:
        name = '{}_{}.png'.format(name,
                                  datetime.now().strftime('%Y%m%d%H%M%S'))
    fig.savefig(os.path.join(path, name))
    log.info('Analysis - Figure was saved in [{}]'.format(
        os.path.join(path, name)))


def show_sample(images, nrows, ncols, save_path=None, logger_name='log'):
    """
    # TODO: #########################################################################
    :param images:
    :param nrows:
    :param ncols:
    :param save_path:
    :param logger_name:
    """
    log = logging.getLogger(logger_name)
    log.info('Analysis - Showing a sample of images: {} rows x {} cols'
             ''.format(nrows, ncols))
    rand_images = random.sample(range(0, len(images)), nrows * ncols)
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(16, 16)
    for i in range(len(rand_images)):
        axs[i].imshow(images[rand_images[i]])
        axs[i].set_title('Example {}'.format(i + 1))
        axs[i].set_axis_off()
    plt.show()
    if save_path:
        save_figure(fig, save_path, 'sample')


def show_sample_by_classes(images, labels, save_path=None, logger_name='log'):
    """
    # TODO: ###########################################################################
    :param images:
    :param labels:
    :param save_path:
    :param logger_name:
    """
    log = logging.getLogger(logger_name)
    unique_labels = set(labels)
    log.info('Analysis - Showing a sample of images by ALL classes:'
             ' {} unique classes were found'.format(len(unique_labels)))
    ncols = 10
    nrows = math.ceil(len(unique_labels) / ncols)
    fig, axs = plt.subplots(nrows, ncols)
    [axi.set_axis_off() for axi in axs.ravel()]
    fig.set_size_inches(16, 16)
    i, j = 0, 0
    for label in unique_labels:
        temp_im = images[labels.index(label)]
        row = i % nrows
        col = j % ncols
        axs[row, col].imshow(temp_im)
        axs[row, col].set_title(
            'Class {}, {}'.format(label, labels.count(label)))
        # axs[row, col].set_axis_off()
        j += 1
        i = i if j % ncols != 0 else i + 1
    plt.show()
    if save_path:
        save_figure(fig, save_path, 'sample_by_classes')


if __name__ == '__main__':
    cfg_log = load_config('./conf/conf.yaml', section='logging')
    cfg_an = load_config('./conf/conf.yaml', section='analysis')
    logger = init_logger(cfg_log, cfg_log['name'])
    ims, labs = load_data(cfg_an['input']['path'])
    # images = np.array(images)
    data_summary(np.array(ims))
    plt.hist(labs, len(set(labs)))
    plt.show()  # TODO: This should be created as done in other functions! Create a fig and manipulate it as preferred
    fig_path = cfg_an['figures']['save_path']
    show_sample(ims, 1, 6, save_path=cfg_an['figures']['save_path'])
    show_sample_by_classes(ims, labs, save_path=cfg_an['figures']['save_path'])



    plt.close('all')
