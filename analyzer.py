#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging


class AnalyzerException(Exception):
    """
    Exception to be raised if an error occurs during a Preprocessing task
    """
    pass


class Analyzer:

    def __init__(self, logger_name='log'):
        self.__logger = logging.getLogger(logger_name)
        pass