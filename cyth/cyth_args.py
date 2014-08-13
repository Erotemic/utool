from __future__ import absolute_import, division, print_function
import utool

DYNAMIC = not utool.get_flag('--nodyn')
WITH_CYTH = not utool.get_flag('--nocyth')
CYTH_WRITE = utool.get_flag('--cyth-write')
