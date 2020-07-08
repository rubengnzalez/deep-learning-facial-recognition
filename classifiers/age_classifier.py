#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging
import os
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from common.utils import create_dir


class AgeClassifier:

    def __init__(self, name, training_path, test_path, training_size,
                 test_size, target_list, cfg, fig_path='figures/',
                 **kwargs):
        self.logger = logging.getLogger(kwargs.get('logger_name')
                                        if kwargs.get('logger_name', False)
                                        else 'log')
        if kwargs.get('force_gpu', True):
            physical_devices = \
                tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.name = name
        self.training_set, self.test_set = \
            self.generate_datasets(training_path, test_path)
        self.training_size, self.test_size = training_size, test_size
        self.target_list = target_list
        self.cfg = cfg
        self.model, self.history = None, None
        self.fig_path = fig_path

    def generate_datasets(self, training_path, test_path):
        """
        It creates Data Generators for data augmentation.
        :param training_path: Path to training set
        :param test_path: Path to test set
        :return: Tuple of generators: (training, test)
        """
        conf = self.cfg['data_augmentation']
        training_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=conf['training_set']['rotation_range'],
            shear_range=conf['training_set']['shear_range'],
            zoom_range=conf['training_set']['zoom_range'],
            horizontal_flip=conf['training_set']['horizontal_flip'])
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        training_set = training_datagen.flow_from_directory(
            training_path,
            target_size=conf['training_set']['target_size'],
            batch_size=conf['training_set']['batch_size'],
            class_mode=conf['training_set']['class_mode'],
            shuffle=conf['training_set']['shuffle'],
            seed=conf['training_set']['seed'])
        test_set = test_datagen.flow_from_directory(
            test_path,
            target_size=conf['test_set']['target_size'],
            batch_size=conf['test_set']['batch_size'],
            class_mode=conf['test_set']['class_mode'],
            shuffle=conf['test_set']['shuffle'],
            seed=conf['test_set']['seed'])
        return training_set, test_set

    def compile(self, summary=True):
        """
        It builds and compiles the deep learning network
        :param summary: Determines whether the model summary is shown or not
        """
        arc_cfg, cmp_cfg = self.cfg['architecture'], self.cfg['compilation']
        # Build architecture
        self.logger.info('{} - Building model'.format(self.name.upper()))
        self.model = Sequential()
        # add first Conv2D layer
        self.model.add(
            Conv2D(filters=arc_cfg['input_layer']['filters'],
                   kernel_size=arc_cfg['input_layer']['kernel_size'],
                   input_shape=arc_cfg['input_layer']['input_shape'],
                   activation=arc_cfg['input_layer']['activation']))
        # check if MaxPool2D layer should be appended
        if arc_cfg['input_layer'].get('max_pooling'):
            self.model.add(MaxPool2D(
                pool_size=arc_cfg['input_layer']['max_pooling']['pool_size']))
        # check if hidden Conv2D are configured and add them if needed
        if arc_cfg.get('hidden_conv') and \
                isinstance(arc_cfg.get('hidden_conv'), list):
            self.add_conv2d_layers(arc_cfg['hidden_conv'])
        # add Flatten layer
        self.model.add(Flatten())
        # check if hidden Dense layers are configured and add them if needed
        if arc_cfg.get('hidden_dense') and \
                isinstance(arc_cfg.get('hidden_dense'), list):
            self.add_dense_layers(arc_cfg['hidden_dense'])
        # Compile model
        self.logger.info('{} - Compiling model'.format(self.name.upper()))
        self.model.compile(optimizer=cmp_cfg['optimizer'],
                           loss=cmp_cfg['loss'],
                           metrics=cmp_cfg['metrics'])
        if summary:
            self.show_summary()

    def train(self):
        """
        It trains the model i.e. it calls the fit() method from model to train
        the network according to the training set generated.
        """
        aug_cfg = self.cfg['data_augmentation']
        self.logger.info('{} - Training model'.format(self.name.upper()))
        training_steps = \
            self.training_size // aug_cfg['training_set']['batch_size'] + 1
        test_steps = self.test_size // aug_cfg['test_set']['batch_size'] + 1
        self.history = self.model.fit(self.training_set,
                                      steps_per_epoch=training_steps,
                                      epochs=self.cfg['training']['epochs'],
                                      validation_data=self.test_set,
                                      validation_steps=test_steps)

    def add_conv2d_layers(self, layers_list):
        """
        It adds Conv2D layers according to the layers list passed as argument.
        Additionally, it adds MaxPool2D layer if needed.
        :param layers_list: list containing configuration of layers
        """
        for layer in layers_list:
            self.model.add(Conv2D(filters=layer['filters'],
                                  kernel_size=layer['kernel_size'],
                                  activation=layer['activation']))
            # check if MaxPool2D layer should be appended
            if layer.get('max_pooling'):
                self.model.add(
                    MaxPool2D(pool_size=layer['max_pooling']['pool_size']))

    def add_dense_layers(self, layers_list):
        """
        It adds Dense layers according to the layers list passed as argument.
        Additionally, it adds Dropout if needed.
        :param layers_list: list containing configuration of layers
        """
        for layer in layers_list:
            self.model.add(Dense(units=layer['units'],
                                 activation=layer['activation']))
            # check if Dropout should be applied
            if layer.get('dropout'):
                self.model.add(Dropout(layer['dropout']))

    def show_summary(self):
        """
        It shows a summary of the model architecture
        """
        self.logger.info('{} - Model summary'.format(self.name.upper()))
        self.logger.info(self.model.summary())

    def plot_confusion_matrix(self, steps, set_name='', save_path=None):
        """
        It plots confusion matrix and classification report.
        :param steps: steps for prediction
        :param set_name: Name of the set to be displayed
        :param save_path: Path to save the figure
        """
        if not save_path:
            save_path = self.fig_path
        y = self.model.predict(self.test_set, steps=steps)
        y_pred = np.argmax(y, axis=1)
        conf_mat = confusion_matrix(self.test_set.classes, y_pred)
        self.logger.info('{} - Confusion Matrix\n{}'.format(self.name,
                                                            conf_mat))
        clas_rep = classification_report(self.test_set.classes,
                                         y_pred,
                                         target_names=self.target_list)
        self.logger.info('{} - Classification Report\n{}'.format(self.name,
                                                                 clas_rep))
        con_mat_norm = np.around(
            conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis],
            decimals=2)
        con_mat_df = pd.DataFrame(con_mat_norm,
                                  index=self.target_list,
                                  columns=self.target_list)
        fig, axs = plt.subplots()
        fig.set_size_inches(8, 8)
        sn.heatmap(con_mat_df, annot=True, cmap=plt.get_cmap('Blues'))
        fig.tight_layout()
        axs.set_ylabel('True')
        axs.set_xlabel('Predicted')
        fig.show()
        if save_path:
            fname = '{}_confusion_matrix'.format(self.name) if not set_name \
                else '{}_confusion_matrix'.format(set_name)
            self.save_figure(fig, fname, save_path)

    def plot_model(self, show_shapes=True, show_layer_names=True):
        """
        It creates a file with a plot of the model/network architecture. It
        shows the shapes and the layers if wanted.
        :param show_shapes: boolean that determines if shapes should be shown
        :param show_layer_names: boolean that determines if layer names should
        be shown
        """
        path = os.path.join(self.fig_path,
                            '{}_model_architecture.png'.format(self.name))
        plot_model(self.model,
                   to_file=path,
                   show_shapes=show_shapes,
                   show_layer_names=show_layer_names)

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
            self.logger.info('Model - Path [{}] does not exist. Creating...'
                             ''.format(path))
            create_dir(path)
        if append_date:
            name = '{}_{}.png'.format(name,
                                      datetime.now().strftime('%Y%m%d%H%M%S'))
        fig.savefig(os.path.join(path, name))
        self.logger.info('Model - Figure was saved in [{}]'.format(
            os.path.join(path, name)))

    def show_training_history(self, set_name='', save_path=None):
        """
        Plot training history based on accuracy and loss
        :param set_name: Name of the set to be displayed
        :param save_path: Path to save the figure
        :return:
        """
        if not save_path:
            save_path = self.fig_path
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)
        title_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
                      'size': 18}
        label_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
                      'size': 14}

        # Accuracy - Train vs. Test
        axs[0].plot(self.history.history['accuracy'], lw=2)
        axs[0].plot(self.history.history['val_accuracy'], lw=2)
        axs[0].set_title('Accuracy History', fontdict=title_font)
        axs[0].set_ylabel('Accuracy', fontdict=label_font)
        axs[0].set_xlabel('Epoch', fontdict=label_font)
        axs[0].legend(['train', 'test'], loc='upper left')

        # Loss - Train vs. Test
        axs[1].plot(self.history.history['loss'], lw=2)
        axs[1].plot(self.history.history['val_loss'], lw=2)
        axs[1].set_title('Loss History', fontdict=title_font)
        axs[1].set_ylabel('Loss', fontdict=label_font)
        axs[1].set_xlabel('Epoch', fontdict=label_font)
        axs[1].legend(['train', 'test'], loc='upper left')
        fig.show()
        if save_path:
            fname = '{}_training_history'.format(self.name) if not set_name \
                else '{}_training_history'.format(set_name)
            self.save_figure(fig, fname, save_path)

    def save_model(self, path):
        """
        It saves the model on disk as well as the weights assigned to the
        neurons in the network
        :param path: path where the model and weight files should be stored
        """
        if not os.path.exists(path):
            self.logger.info('Model - Path [{}] does not exist. Creating...'
                             ''.format(path))
            create_dir(path)
        self.model.save(os.path.join(path, self.name + '_model.h5'))
        self.model.save_weights(os.path.join(path, self.name + '_weights.h5'))

    def load_model(self, model_path, weights_path):
        """
        It loads the model and the weights from file. It corresponds to a
        previuosly trained deep learning model
        :param model_path: path to the file with the trained model
        :param weights_path: path to the file with the trained weights
        """
        self.model = load_model(model_path)
        self.model.load_weights(weights_path)

    # TODO: PREDICT WITH OTHER IMAGES TO TEST THE RESPONSE OF THE MODEL
