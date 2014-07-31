#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import os
import sys
argvx = sys.argv.find('profiler.py')
argv = sys.argv[argvx:]
cmd = 'profiler.sh ' + ' '.join(argv)
print(cmd)
os.system(cmd)
