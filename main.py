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
    model_name = 'test-model'
    training_size = 18966
    test_size = 4742
    model_cfg = {
        'data_augmentation': {
            'training_set': {
                'rotation_range': 0.3,
                'shear_range': 0.1,
                'zoom_range': [0.90, 1.2],
                'horizontal_flip': True,
                'target_size': (64, 64),
                'batch_size': 32,
                'class_mode': 'categorical',
                'shuffle': True,
                'seed': 42
            },
            'test_set': {
                'target_size': (64, 64),
                'batch_size': 32,
                'class_mode': 'categorical',
                'shuffle': False,
                'seed': 42
            }
        },
        'architecture': {
            'input_layer': {
                'filters': 32,
                'kernel_size': (3, 3),
                'input_shape': (64, 64, 3),
                'activation': 'relu',
                'max_pooling': {'pool_size': (2, 2)}
            },
            'hidden_conv': [
                {'filters': 32,
                 'kernel_size': (3, 3),
                 'activation': 'relu',
                 'max_pooling': {'pool_size': (2, 2)}
                 },
                {'filters': 32,
                 'kernel_size': (3, 3),
                 'activation': 'relu',
                 'max_pooling': {'pool_size': (2, 2)}
                 },
                {'filters': 32,
                 'kernel_size': (3, 3),
                 'activation': 'relu',
                 'max_pooling': {'pool_size': (2, 2)}
                 }
            ],
            'hidden_dense': [
                {'units': 128, 'activation': 'relu', 'dropout': 0.3},
                {'units': 64, 'activation': 'relu', 'dropout': 0.3},
                {'units': 6, 'activation': 'softmax'},
            ]
        },
        'compilation': {
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        },
        'training': {
            'epochs': 2
        }
    }

    cnn = AgeClassifier(model_name,
                        train_path,
                        test_path,
                        training_size,
                        test_size,
                        cfg['data']['classes_list'],
                        batch_size=32,
                        fig_path=fig_full_path)
    cnn.compile()
    cnn.plot_model()
    cnn.train()
    cnn.show_training_history()
    cnn.plot_confusion_matrix((4742 // 32 + 1))
    cnn.save_model(model_full_path)
