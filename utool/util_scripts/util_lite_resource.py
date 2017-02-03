#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def print_mem_usage():
    import psutil
    import os
    nBytes = psutil.Process(os.getpid()).get_memory_info().rss
    nMegaBytes =  nBytes / (2.0 ** 20)
    print(' * current_memory = %.2f MB' % nMegaBytes)
