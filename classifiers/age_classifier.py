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

    def __init__(self, training_path, test_path, target_list,
                 fig_path='figures/', **kwargs):
        self.logger = logging.getLogger(kwargs.get('logger_name')
                                        if kwargs.get('logger_name', False)
                                        else 'log')
        if kwargs.get('force_gpu', True):
            physical_devices = \
                tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.training_set, self.test_set = \
            self.generate_datasets(training_path, test_path)
        self.target_list = target_list
        self.model, self.history = None, None
        self.fig_path = fig_path
        self.name = 'BLABLABLA'

    @staticmethod
    def generate_datasets(training_path, test_path):
        """
        It creates Data Generators for data augmentation.
        :param training_path: Path to training set
        :param test_path: Path to test set
        :return: Tuple of generators: (training, test)
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
                                                    shuffle=False,
                                                    seed=42)
        return training_set, test_set

    def show_summary(self):
        """
        It shows a summary of the model architecture
        """
        self.logger.info(self.model.summary())

    def compile(self, summary=True):
        """
        It builds and compiles the deep learning network
        :param summary: Determines whether the model summary is shown or not
        """
        # Build architecture
        self.model = Sequential()
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3),
                   activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        # self.model.add(
        #     Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        # self.model.add(MaxPool2D(pool_size=(2, 2)))
        # self.model.add(
        #     Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        # self.model.add(MaxPool2D(pool_size=(2, 2)))
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
        It trains the model i.e. it calls the fit() method from model to train
        the network according to the training set generated.
        """
        training_steps = 18966 // 32 + 1
        test_steps = 4742 // 32 + 1
        self.history = self.model.fit(self.training_set,
                                      steps_per_epoch=training_steps,
                                      epochs=1,
                                      validation_data=self.test_set,
                                      validation_steps=test_steps)

    def plot_confusion_matrix(self, steps, target_names,
                              set_name='', save_path=None):
        """
        It plots confusion matrix and classification report.
        :param steps: steps for prediction
        :param target_names: list of classes
        :param set_name: Name of the set to be displayed
        :param save_path: Path to save the figure
        """
        if not save_path:
            save_path = self.fig_path
        y = self.model.predict(self.test_set, steps=steps)
        y_pred = np.argmax(y, axis=1)
        conf_mat = confusion_matrix(self.test_set.classes, y_pred)
        self.logger.info('Confusion Matrix\n%s' % conf_mat)
        clas_rep = classification_report(self.test_set.classes,
                                         y_pred,
                                         target_names=target_names)
        self.logger.info('Classification Report\n%s' % clas_rep)
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
        :param file_path: path where the file will be created
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
        :param model_path:
        :param weights_path:
        """
        self.model = load_model(model_path)
        self.model.load_weights(weights_path)

    # TODO: PREDICT WITH OTHER IMAGES TO TEST THE RESPONSE OF THE MODEL
