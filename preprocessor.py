#! /usr/bin/env python
# -*- coding: utf-8 -*-
from common.utils import load_config, init_logger



if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    logger = init_logger(cfg['logging'], cfg.get('name', 'log'))
    logger.info(cfg)
