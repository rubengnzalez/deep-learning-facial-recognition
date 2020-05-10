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
    Function to load data from a given data path. It expects the data to be
    already sorted out in their corresponding class folders
    :param data_path: Path to data
    :param logger_name: Name of Logger instance
    :return: tuple (images, labels)
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
    Function that prints a summary of the data loaded. It will provide
    information like shape of the dataset, dimension, total bytes it uses,
    flags...
    :param np_data: Data in a NumPy Array
    :param flags: it determines if Flags should be displayed of not
    :param logger_name: Name of Logger instance
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
    Function that saves the figure passed as param in the path given.
    :param fig: Figure object that will be saved as image
    :param path: Target path
    :param name: Name of the image
    :param append_date: Boolean that specifies if date should be appended
    to image name
    :param logger_name: Name of Logger instance
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


def show_data_distribution(labels, save_path=None, logger_name='log'):
    """
    Function that shows a histogram regarding to data distribution by classes
    :param labels: Array containing 1 label per image
    :param save_path: Path to save the figure
    :param logger_name: Name of Logger instance
    """
    log = logging.getLogger(logger_name)
    unique_labels = set(labels)
    log.info('Analysis - Showing data distribution histogram: {} items and '
             ' {} unique classes were found'.format(len(labels),
                                                    len(unique_labels)))

    fig, ax = plt.subplots()
    ax.hist(labels, len(unique_labels))
    ax.set_title('Data Distribution')
    fig.show()
    if save_path:
        save_figure(fig, save_path, 'sample_by_classes')


def show_sample(images, nrows, ncols, save_path=None, logger_name='log'):
    """
    Function that shows a sample of the data loaded. This is just to verify
    that data has been loaded properly. Number of rows and columns of the
    figure will be passed as argument.
    Images displayed will be chosen randomly in each execution
    :param images: Array containing images
    :param nrows: rows to be displayed
    :param ncols: cols to be displayed in each row
    :param save_path: Path to save the figure
    :param logger_name: Name of Logger instance
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
    fig.show()
    if save_path:
        save_figure(fig, save_path, 'sample')


def show_sample_by_classes(images, labels, save_path=None, logger_name='log'):
    """
    Function that shows a sample of the data, with 1 image per class.
    It will display a randomly chosen image for each class
    :param images: Array containing images
    :param labels: Array containing labels
    :param save_path: Path to save the figure
    :param logger_name: Name of Logger instance
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
    fig.show()
    if save_path:
        save_figure(fig, save_path, 'sample_by_classes')


if __name__ == '__main__':
    cfg_log = load_config('./conf/conf.yaml', section='logging')
    cfg_an = load_config('./conf/conf.yaml', section='analysis')
    logger = init_logger(cfg_log, cfg_log['name'])
    ims, labs = load_data(cfg_an['input']['path'])
    data_summary(np.array(ims))
    fig_path = cfg_an['figures']['save_path']
    show_data_distribution(labs, save_path=fig_path)
    show_sample(
        ims, cfg_an['sample_rows'], cfg_an['sample_cols'], save_path=fig_path)
    show_sample_by_classes(ims, labs, save_path=fig_path)
    plt.close('all')
