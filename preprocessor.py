#! /usr/bin/env python
# -*- coding: utf-8 -*-
from common.utils import load_config



if __name__ == '__main__':
    cfg = load_config('./conf/conf.yaml')
    print(cfg)
