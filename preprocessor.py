#! /usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ruben Gonzalez Lozano'
__email__ = '100284010@alumnos.uc3m.es'

import logging
import os

from common.utils import remove_dir, create_dir, copy_file


class PreprocessorException(Exception):
    """
    Exception to be raised if an error occurs during a Preprocessing task
    """
    pass


class Preprocessor:

    def __init__(self, logger_name='log'):
        self.__logger = logging.getLogger(logger_name)
        pass

    def classify_data_in_folders(self, input_path, file_name_format, criteria,
                                 output_path, classes):
        """
    # TODO:  ############################################################################
        :param input_path:
        :param file_name_format:
        :param criteria:
        :param output_path:
        :param classes:
        :return:
        """
        self.__logger.info(
            'Classifying images into their corresponding folders according to '
            'criteria "{crit}" and the following classes distribution: {clas}'
            ''. format(crit=criteria, clas=classes))
        full_input_path = os.path.abspath(input_path)
        full_output_path = os.path.abspath(output_path)
        if os.path.exists(full_input_path) and \
                os.path.exists(full_output_path):
            crit_path = os.path.join(full_output_path, criteria)
            crit_index = file_name_format.split('_').index(criteria)
            if os.path.exists(crit_path):
                # just in case it exists and contains previous executions data
                remove_dir(crit_path)
            else:
                create_dir(crit_path)
            for r, d, f in os.walk(full_input_path):
                for file in f:
                    if os.path.isfile(os.path.join(r, file)):
                        crit_value = file.split('_')[crit_index]
                        ext = '.' + file.split('.')[-1]
                        for k, v in classes.items():
                            if v[0] <= int(crit_value) <= v[1]:
                                if not os.path.exists(
                                        os.path.join(crit_path, str(k))):
                                    create_dir(os.path.join(crit_path, str(k)))
                                copy_file(os.path.join(r, file),
                                          os.path.join(
                                              crit_path,
                                              str(k),
                                              file.split('.')[0] + ext))
        else:
            raise PreprocessorException(
                'An error occurred while classifying images in their '
                'corresponding folders')


if __name__ == '__main__':
    Preprocessor()
