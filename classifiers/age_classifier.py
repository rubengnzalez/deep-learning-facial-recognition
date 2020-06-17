#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from common.utils import create_dir


class AgeClassifier:

    def __init__(self, training_path, test_path, fig_path='figures/',
                 **kwargs):
        self.logger = logging.getLogger(kwargs.get('logger_name')
                                        if kwargs.get('logger_name', False)
                                        else 'log')
        if kwargs.get('force_gpu', True):
            physical_devices = \
                tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.training_set, self.test_set = \
            self.generate_datasets(training_path, test_path)
        self.model, self.history = None, None
        self.fig_path = fig_path

    @staticmethod
    def generate_datasets(training_path, test_path):
        """
        # TODO: ###################################################################
        :param training_path:
        :param test_path:
        :return:
        """
        training_datagen = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=0.3,
                                              shear_range=0.1,
                                              zoom_range=[0.90, 1.2],
                                              horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        training_set = \
            training_datagen.flow_from_directory(training_path,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True,
                                                 seed=42)
        test_set = test_datagen.flow_from_directory(test_path,
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    seed=42)
        return training_set, test_set

    def show_summary(self):
        """
        # TODO: ###################################################################
        """
        self.logger.info(self.model.summary())

    def compile(self, summary=True):
        """
        # TODO: ###################################################################
        :param summary:
        """
        # Build architecture
        self.model = Sequential()
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3),
                   activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=6, activation='softmax'))
        # Compile model
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        if summary:
            self.show_summary()

    def train(self):
        """
        # TODO: ####################################################################
        """
        self.history = self.model.fit(self.training_set,
                                      steps_per_epoch=(18966 // 32),
                                      epochs=25,
                                      validation_data=self.test_set,
                                      validation_steps=(4742 // 32))

    def plot_model(self, file_path, show_shapes=True, show_layer_names=True):
        """
        # TODO: #####################################################################
        :param file_path:
        :param show_shapes:
        :param show_layer_names:
        :return:
        """
        plot_model(self.model,
                   to_file=file_path,
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
        # TODO: ###################################################################
        :param set_name: Name of the set to be displayed
        :param save_path: Path to save the figure
        :return:
        """
        if not save_path:
            save_path = self.fig_path
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(8, 8)
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
            fname = 'training_history' if not set_name \
                else '{}_training_history'.format(set_name)
            self.save_figure(fig, fname, save_path)

    # TODO: SAVE MODEL, WEIGHTS AND HISTORY (DICT, with pickle) ON DISK FOR LATER WORKING SESSIONS
    def save_model(self, path, name):
        """
        # TODO: ####################################################################
        :param path:
        :param name:
        """
        if not os.path.exists(path):
            self.logger.info('Model - Path [{}] does not exist. Creating...'
                             ''.format(path))
            create_dir(path)
        self.model.save(os.path.join(path, name + '_model.h5'))
        self.model.save_weights(os.path.join(path, name + '_weights.h5'))

    # TODO: LOAD MODEL AND WEIGHTS FROM A GIVEN PATH!!
    def load_model(self, model_path, weights_path):
        """
        # TODO: ####################################################################
        :param model_path:
        :param weights_path:
        """
        self.model = load_model(model_path)
        self.model.load_weights(weights_path)

    # TODO: PREDICT WITH OTHER IMAGES TO TEST THE RESPONSE OF THE MODEL
