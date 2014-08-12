"""
python -c "import doctest, cyth; print(doctest.testmod(cyth.cyth_decorators))"
"""
from __future__ import absolute_import, division, print_function
import utool

REGISTERED_FUNCS = utool.ddict(list)


def register(func):
    global REGISTERED_FUNCS
    modname = func.func_globals['__name__']
    REGISTERED_FUNCS[modname].append(func)
    return func


def get_registered_funcs(modname):
    return REGISTERED_FUNCS[modname]
