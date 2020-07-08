#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import os
from analyzer import Analyzer
from common.utils import load_config, init_logger
from preprocessor import Preprocessor
from classifiers.classifier import Classifier

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
    fig_full_path = os.path.abspath(cfg['model']['figures_path'])
    model_full_path = os.path.abspath(cfg['model']['models_path'])
    weights_full_path = os.path.abspath(cfg['model']['weights_path'])
    training_size = 18966
    test_size = 4742

    cnn = Classifier(train_path,
                     test_path,
                     training_size,
                     test_size,
                     cfg['data']['classes_list'],
                     cfg['model'],
                     fig_path=fig_full_path)
    cnn.compile()
    cnn.plot_model()
    cnn.train()
    cnn.show_training_history()
    cnn.plot_confusion_matrix((4742 // 32 + 1))
    cnn.save_model(model_full_path)
