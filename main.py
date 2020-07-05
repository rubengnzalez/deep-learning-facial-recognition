#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import os

from analyzer import Analyzer
from common.utils import load_config, init_logger
from preprocessor import Preprocessor
from classifiers.age_classifier import AgeClassifier

if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    init_logger(cfg['logging'], cfg['logging']['name'])
    # ##############################################################
    # #                      PREPROCESSING                         #
    # ##############################################################
    # prep = Preprocessor(cfg['data']['raw_path'],
    #                     cfg['preprocess']['criteria'],
    #                     cfg['data']['file_name_format'],
    #                     cfg['data']['classes_list'],
    #                     cfg['preprocess']['classes_ranges'],
    #                     cfg['preprocess']['dest_path'])
    # prep.run(test_size=cfg['preprocess']['test_size'],
    #          random_state=cfg['preprocess']['random_state'],
    #          shuffle=cfg['preprocess']['shuffle'],
    #          stratify=True)

    ##############################################################
    #                         ANALYSIS                           #
    ##############################################################
    # TODO: Analysis - Complete dataset, ignoring classes

    # tr_path = os.path.join(cfg['data']['sorted_path'], 'training')
    # te_path = os.path.join(cfg['data']['sorted_path'], 'test')
    # anlz = Analyzer(tr_path,
    #                 te_path,
    #                 cfg['data']['classes_list'],
    #                 cfg['analysis']['figures_path'])
    # anlz.run()

    ##############################################################
    #                        CNN MODEL                           #
    ##############################################################
    train_path = os.path.join(cfg['data']['sorted_path'], 'training')
    test_path = os.path.join(cfg['data']['sorted_path'], 'test')
    fig_full_path = os.path.abspath(cfg['analysis']['figures_path'])
    model_full_path = os.path.abspath('classifiers/models')

    cnn = AgeClassifier(train_path, test_path, fig_path=fig_full_path)
    cnn.compile()
    cnn.plot_model(os.path.join(fig_full_path, 'model_architecture.png'))
    cnn.train()
    cnn.show_training_history()
    cnn.plot_confusion_matrix((4742 // 32), cfg['data']['classes_list'])
    cnn.save_model(model_full_path)






























    # TODO:
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # from tensorflow.keras import Sequential
    # from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, \
    #     Dense
    # import tensorflow as tf
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
    # training_datagen = ImageDataGenerator(rescale=1. / 255,
    #                                       rotation_range=0.3,
    #                                       shear_range=0.1,
    #                                       zoom_range=[0.90, 1.2],
    #                                       horizontal_flip=True)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # training_path = '/home/ruben/workspace/tfg/deep-learning-facial-recognition/data/age/training'
    # test_path = '/home/ruben/workspace/tfg/deep-learning-facial-recognition/data/age/test'
#
    # training_set = training_datagen.flow_from_directory(training_path,
    #                                                     target_size=(64, 64),
    #                                                     batch_size=32,
    #                                                     class_mode='categorical',
    #                                                     shuffle=True,
    #                                                     seed=42)
#
    # test_set = test_datagen.flow_from_directory(test_path,
    #                                             target_size=(64, 64),
    #                                             batch_size=32,
    #                                             class_mode='categorical',
    #                                             shuffle=True,
    #                                             seed=42)
    # age_classifier = Sequential()
    # age_classifier.add(
    #     Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3),
    #            activation='relu'))
    # age_classifier.add(MaxPool2D(pool_size=(2, 2)))
    # age_classifier.add(
    #     Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    # age_classifier.add(MaxPool2D(pool_size=(2, 2)))
    # age_classifier.add(
    #     Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # age_classifier.add(MaxPool2D(pool_size=(2, 2)))
    # age_classifier.add(Flatten())
    # age_classifier.add(Dense(units=128, activation='relu'))
    # age_classifier.add(Dropout(0.3))
    # age_classifier.add(Dense(units=64, activation='relu'))
    # age_classifier.add(Dropout(0.3))
    # age_classifier.add(Dense(units=6, activation='softmax'))
    # age_classifier.compile(optimizer='adam', loss='categorical_crossentropy',
    #                        metrics=['accuracy'])
    # hist = age_classifier.fit(training_set,
    #                           steps_per_epoch=(18966 // 32),
    #                           epochs=25,
    #                           validation_data=test_set,
    #                           validation_steps=(4742 // 32))
#
#
    # # TODO: PRINT MODEL SUMMARY (LAYERS / ARCHITECTURE)
    # from tensorflow.python.keras.utils.vis_utils import plot_model
    # print(age_classifier.summary())
    # plot_model(age_classifier, to_file='figures/model_plot.png',
    #            show_shapes=True, show_layer_names=True)
#
#
    # # TODO: SAVE HISTORY OF TRAINING PROCESS AND SHOW SOME GRAPHS ON LOSS AND ACCURACY
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2)
    # fig.set_size_inches(8, 8)
    # title_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
    #               'size': 18}
    # label_font = {'family': 'serif', 'color': 'black', 'weight': 'normal',
    #               'size': 14}
    # # Accuracy - Train vs. Test
    # axs[0].plot(hist.history['accuracy'], lw=2)
    # axs[0].plot(hist.history['val_accuracy'], lw=2)
    # axs[0].set_title('Accuracy History', fontdict=title_font)
    # axs[0].set_ylabel('Accuracy', fontdict=label_font)
    # axs[0].set_xlabel('Epoch', fontdict=label_font)
    # axs[0].legend(['train', 'test'], loc='upper left')
    # # Loss - Train vs. Test
    # axs[1].plot(hist.history['loss'], lw=2)
    # axs[1].plot(hist.history['val_loss'], lw=2)
    # axs[1].set_title('Loss History', fontdict=title_font)
    # axs[1].set_ylabel('Loss', fontdict=label_font)
    # axs[1].set_xlabel('Epoch', fontdict=label_font)
    # axs[1].legend(['train', 'test'], loc='upper left')
    # fig.show()


    # TODO: SAVE MODEL, WEIGHTS AND HISTORY (DICT, with pickle) ON DISK FOR LATER WORKING SESSIONS


    # TODO: PREDICT WITH OTHER IMAGES TO TEST THE RESPONSE OF THE MODEL
