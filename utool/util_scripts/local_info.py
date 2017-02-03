#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" flake8: noqa """
from __future__ import absolute_import, division, print_function
import utool
import psutil  # NOQA
import multiprocessing  # NOQA
import sys  # NOQA
import os  # NOQA
import platform
import site  # NOQA

if __name__ == '__main__':
    """
    CommandLine:
        python utool/util_scripts/local_info.py
    """

    for name in filter(lambda x: not x.startswith('_'), dir(platform)):
        attr = getattr(platform, name)
        if hasattr(attr, '__call__'):
            if attr.func_code.co_argcount == 0:
                utool.printvar2('platform.' + name + '()', typepad=14)

    utool.printvar2('multiprocessing.cpu_count()')
    utool.printvar2('sys.platform')
    utool.printvar2('os.getcwd()')
    utool.printvar2('utool.is64bit_python()')
    utool.printvar2('sys.maxint')

    print('')
    print('Python Site:')
    utool.printvar2('site.getsitepackages()')
    utool.printvar2('site.getusersitepackages()')

    print('')
    print('Memory Status:')
    print(utool.get_memstats_str())

    try:
        #import psutil
        #print('')
        #print('PSUTIL CPUS')
        #print('')
        #utool.printvar2('psutil.cpu_times()')
        #utool.printvar2('psutil.NUM_CPUS')
        #print('')
        #print('PSUTIL MEMORY')
        #print('')
        #utool.printvar2('psutil.virtual_memory()')
        #utool.printvar2('psutil.swap_memory()')
        #print('')
        #print('PSUTIL DISK')
        #print('')
        #utool.printvar2('psutil.disk_partitions()')
        #utool.printvar2('psutil.disk_usage("/")')
        #utool.printvar2('psutil.disk_io_counters()')
        #print('')
        #print('PSUTIL NETWORK')
        #print('')
        #utool.printvar2('psutil.net_io_counters(pernic=True)')
        #print('')
        #print('PSUTIL MISC')
        #print('')
        #utool.printvar2('psutil.get_users()')
        #utool.printvar2('psutil.get_boot_time()')
        #utool.printvar2('psutil.get_pid_list()')

        #psutil.test()
        pass
    except ImportError:
        print('psutil not installed')

    try:
        import resource
        utool.rrr()
        used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print('[parallel] Max memory usage: %s' % utool.byte_str2(used_memory))
    except ImportError:
        print('no module resources (doesnt exist on win32)')

    try:
        import cv2  # NOQA
        utool.printvar2('cv2.__version__')
    except ImportError:
        print('cv2 is not installed')
