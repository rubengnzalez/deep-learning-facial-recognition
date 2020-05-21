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
from common.utils import init_logger, load_config, create_dir, get_pos,\
    get_value


class AnalysisError(Exception):
    """
    Exception to be raised if an error occurs during a Preprocessing task
    """
    pass


class Analyzer:

    def __init__(self, train_path, test_path, classes, fig_path='figures/',
                 **kwargs):
        self.logger = logging.getLogger(kwargs.get('logger_name')
                                        if kwargs.get('logger_name', False)
                                        else 'log')
        self.classes = classes
        self.fig_path = fig_path
        self.train_imgs, self.train_labls = self.load_dataset(train_path)
        self.test_imgs, self.test_labls = self.load_dataset(test_path)

    def load_dataset(self, data_path):
        """
        Function to load data from a given data path. It expects the data to be
        already sorted out in their corresponding class folders
        :param data_path: Path to data
        :return: tuple (images, labels)
        """
        self.logger.info(
            'Analysis - Loading data from path [{}]'.format(data_path))
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
                lab.append(get_pos(d, self.classes))
        self.logger.info('Analysis - Loaded {} images and {} different classes'
                         ''.format(len(im), len(set(lab))))
        return im, lab

    def show_data_summary(self, data, set_name='', flags=True):
        """
        Function that prints a summary of the data loaded. It will provide
        information like shape of the dataset, dimension, total bytes it uses,
        flags...
        :param data: Data to be summarized
        :param set_name: Name of the set to be summarized
        :param flags: it determines if Flags should be displayed of not
        """
        np_data = np.array(data)
        summary = 'Analysis - {} - Data Summary [Np_Dim: {}, Np_Bytes: {}, ' \
                  'Np_ItemSize: {}, Shape: {}]'.format(set_name,
                                                       np_data.ndim,
                                                       np_data.nbytes,
                                                       np_data.itemsize,
                                                       np_data.shape)
        self.logger.info(summary)
        if flags:
            self.logger.info('Analysis - {} - Data Flags [{}]'.format(
                set_name, str(np_data.flags).replace('\n', '')))

    def show_train_data_summary(self, flags=True):
        """
        Wrapper for show_data_summary to show training data summary
        :param flags: it determines if Flags should be displayed of not
        """
        self.show_data_summary(self.train_imgs, 'TRAINING', flags)

    def show_test_data_summary(self, flags=True):
        """
        Wrapper for show_data_summary to show test data summary
        :param flags: it determines if Flags should be displayed of not
        """
        self.show_data_summary(self.test_imgs, 'TEST', flags)

    def save_figure(self, fig, name, path=None, append_date=True):
        """
        Function that saves the figure passed as param in the path given.
        :param fig: Figure object that will be saved as image
        :param name: Name of the image
        :param path: Target path
        :param append_date: Boolean that specifies if date should be appended
        to image name
        """
        if not path:
            path = self.fig_path
        if not os.path.exists(path):
            self.logger.info('Analysis - Path [{}] does not exist. Creating...'
                             ''.format(path))
            create_dir(path)
        if append_date:
            name = '{}_{}.png'.format(name,
                                      datetime.now().strftime('%Y%m%d%H%M%S'))
        fig.savefig(os.path.join(path, name))
        self.logger.info('Analysis - Figure was saved in [{}]'.format(
            os.path.join(path, name)))

    def show_data_distribution(self, labels, set_name='', save_path=None):
        """
        Function that shows a Bar Chart regarding to data distribution by
        classes
        :param labels: Array containing 1 label per image
        :param set_name: Name of the set to be displayed
        :param save_path: Path to save the figure
        """
        if not save_path:
            save_path = self.fig_path
        unique_labels = set(labels)
        self.logger.info('Analysis - {} - Showing data distribution histogram:'
                         ' {} items and {} unique classes were found'
                         ''.format(set_name, len(labels), len(unique_labels)))
        fig, ax = plt.subplots()
        title_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
                      'size': 18}
        label_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
                      'size': 14}
        labs, counts = np.unique(labels, return_counts=True)
        ax.bar(labs, counts, align='center')
        fig.gca().set_xticks(labs)
        title = 'Data Distribution' if not set_name \
            else 'Data Distribution - {}'.format(set_name)
        ax.set_title(title, fontdict=title_font)
        ax.set_ylabel('Examples', fontdict=label_font)
        ax.set_xlabel('Classes', fontdict=label_font)
        ax.set_xticklabels(
            self.classes, rotation=45, fontdict={'family': 'serif'})
        fig.show()
        if save_path:
            fname = 'classes_distribution' if not set_name \
                else '{}_classes_distribution'.format(set_name)
            self.save_figure(fig, fname, save_path)

    def show_train_data_distribution(self, save_path=None):
        """
        Wrapper for show_data_distribution to show training data distribution
        :param save_path: Path to save the figure
        """
        self.show_data_distribution(self.train_labls, 'TRAINING', save_path)

    def show_test_data_distribution(self, save_path=None):
        """
        Wrapper for show_data_distribution to show test data distribution
        :param save_path: Path to save the figure
        """
        self.show_data_distribution(self.test_labls, 'TEST', save_path)

    def show_train_test_distribution(self, save_path=None):
        """
        Displays a Stacked Bar Chart with Training and Test distributions
        :param save_path: Path to save the figure
        :return:
        """
        if not save_path:
            save_path = self.fig_path
        self.logger.info('Analysis - Showing data distribution: Train ({}) '
                         'vs. Test ({})'.format(len(self.train_labls),
                                                len(self.test_labls)))
        train_lbls, train_counts = \
            np.unique(self.train_labls, return_counts=True)
        test_lbls, test_counts = np.unique(self.test_labls, return_counts=True)
        fig, ax = plt.subplots()
        ind = np.arange(len(self.classes))  # the x locations for the groups
        width = 0.5
        p1 = ax.bar(ind, train_counts, width)
        p2 = ax.bar(ind, test_counts, width, bottom=train_counts)
        title_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
                      'size': 18}
        ax.set_title('Data Distribution - Train vs. Test', fontdict=title_font)
        label_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
                      'size': 14}
        ax.set_ylabel('Examples', fontdict=label_font)
        ax.set_xlabel('Classes', fontdict=label_font)
        ax.set_xticklabels(self.classes, rotation=45,
                           fontdict={'family': 'serif'})
        fig.gca().set_xticks(train_lbls)
        ax.legend((p1[0], p2[0]), ('Train', 'Test'))
        fig.show()
        if save_path:
            self.save_figure(fig, 'train_vs_test_distrib', save_path)

    def show_sample(self, images, nrows, ncols, set_name='', save_path=None):
        """
        Function that shows a sample of the data loaded. This is just to verify
        that data has been loaded properly. Number of rows and columns of the
        figure will be passed as argument.
        Images displayed will be chosen randomly in each execution
        :param images: Array containing images
        :param nrows: rows to be displayed
        :param ncols: cols to be displayed in each row
        :param set_name: Name of the set to be displayed
        :param save_path: Path to save the figure
        """
        if not save_path:
            save_path = self.fig_path
        self.logger.info('Analysis - {} - Showing a sample of images: {} rows '
                         'x {} cols'.format(set_name, nrows, ncols))
        rand_images = random.sample(range(0, len(images)), nrows * ncols)
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(16, 16)
        for i in range(len(rand_images)):
            axs[i].imshow(images[rand_images[i]])
            axs[i].set_title('Example {}'.format(i + 1))
            axs[i].set_axis_off()
        fig.show()
        if save_path:
            fname = 'sample' if not set_name \
                else '{}_sample'.format(set_name)
            self.save_figure(fig, fname, save_path)

    def show_train_sample(self, nrows, ncols, save_path=None):
        """
        Wrapper for show_sample to show training data samples
        :param nrows: rows to be displayed
        :param ncols: cols to be displayed in each row
        :param save_path: Path to save the figure
        """
        self.show_sample(self.train_imgs, nrows, ncols, 'TRAINING', save_path)

    def show_test_sample(self, nrows, ncols, save_path=None):
        """
        Wrapper for show_sample to show test data samples
        :param nrows: rows to be displayed
        :param ncols: cols to be displayed in each row
        :param save_path: Path to save the figure
        """
        self.show_sample(self.test_imgs, nrows, ncols, 'TEST', save_path)

    def show_sample_by_classes(self, images, labels, classes, set_name='',
                               save_path=None):
        """
        Function that shows a sample of the data, with 1 image per class.
        It will display a randomly chosen image for each class
        :param images: Array containing images
        :param labels: Array containing labels
        :param classes: list of known classes ordered
        :param set_name: Name of the set to be displayed
        :param save_path: Path to save the figure
        """
        if not save_path:
            save_path = self.fig_path
        unique_labels = set(labels)
        self.logger.info(
            'Analysis - {} - Showing sample of images by ALL classes: {} '
            'unique classes were found'.format(set_name, len(unique_labels)))
        ncols = min(10, len(unique_labels))
        nrows = math.ceil(len(unique_labels) / ncols)
        fig, axs = plt.subplots(nrows, ncols)
        [axi.set_axis_off() for axi in axs.ravel()]
        fig.set_size_inches(16, 16)
        i, j = 0, 0
        for label in unique_labels:
            temp_im = images[labels.index(label)]
            row = i % nrows
            col = j % ncols
            if nrows > 1:
                axs[row, col].imshow(temp_im)
                axs[row, col].set_title('Class {}:{}, {}'.format(
                    label,
                    get_value(label, classes).upper(),
                    labels.count(label)))
            else:
                axs[col].imshow(temp_im)
                axs[col].set_title('Class {}:{}, {}'.format(
                    label,
                    get_value(label, classes).upper(),
                    labels.count(label)))
            j += 1
            i = i if j % ncols != 0 else i + 1
        fig.show()
        if save_path:
            fname = 'sample_by_classes' if not set_name \
                else '{}_sample_by_classes'.format(set_name)
            self.save_figure(fig, fname, save_path)

    def show_train_sample_by_classes(self, save_path=None):
        """
        Wrapper for show_sample_by_classes to show train data samples by class
        :param save_path: Path to save the figure
        """
        self.show_sample_by_classes(self.train_imgs, self.train_labls,
                                    self.classes, 'TRAINING', save_path)

    def show_test_sample_by_classes(self, save_path=None):
        """
        Wrapper for show_sample_by_classes to show test data samples by class
        :param save_path: Path to save the figure
        """
        self.show_sample_by_classes(self.test_imgs, self.test_labls,
                                    self.classes, 'TEST', save_path)

    @staticmethod
    def close_plot(fig=None):
        """
        Closes figure passed as argument. If no argument is given or a string
        'all' is provided, all plots will be closed
        :param fig: figure to be closed. Use 'all' to close all plots
        """
        if (isinstance(fig, str) and fig == 'all') or not fig:
            plt.close('all')
        else:
            plt.close(fig)


if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    init_logger(cfg['logging'], cfg['logging']['name'])
    tr_path = os.path.join(cfg['data']['sorted_path'], 'training')
    te_path = os.path.join(cfg['data']['sorted_path'], 'test')
    anlz = Analyzer(tr_path,
                    te_path,
                    cfg['data']['classes_list'],
                    cfg['analysis']['figures_path'])
    # Summaries
    anlz.show_train_data_summary()
    anlz.show_test_data_summary()
    # Individual data distributions by classes
    anlz.show_train_data_distribution()
    anlz.show_test_data_distribution()
    # Training vs Test Data distributions by classes
    anlz.show_train_test_distribution()
    # Some samples for both subsets to verify tehy were loaded properly
    anlz.show_train_sample(1, 4)
    anlz.show_test_sample(1, 4)
    # Show sample with 1 image/class for each subset (training and test)
    anlz.show_train_sample_by_classes()
    anlz.show_test_sample_by_classes()
    # Close plots
    anlz.close_plot('all')
