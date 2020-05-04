#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'


class ConfigurationException(Exception):
    """
    Exception to be raised in case that Configuration is not loaded properly
    """
    pass


class PreprocessingException(Exception):
    """
    Exception to be raised if an error occurs during a Preprocessing task
    """
    pass


class AnalysisException(Exception):
    """
    Exception to be raised if an error occurs during a Preprocessing task
    """
    pass