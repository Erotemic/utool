# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import types
import six
import sys

DEBUG        = '--debug' in sys.argv
DEBUG2       = '--debug2' in sys.argv
NO_ASSERTS   = '--no-assert' in sys.argv
QUIET        = '--quiet' in sys.argv
SILENT       = '--silent' in sys.argv
SAFE         = '--safe' in sys.argv
STRICT       = '--strict' not in sys.argv
REPORT       = '--report' not in sys.argv
TRACE        = '--trace' in sys.argv
SUPER_STRICT = '--super-strict' in sys.argv or '--superstrict' in sys.argv
VERBOSE      = '--verbose' in sys.argv or '--verb' in sys.argv
VERYVERBOSE  = '--very-verbose' in sys.argv or '--veryverbose' in sys.argv
NO_INDENT    = '--noindent' in sys.argv or '--no-indent' in sys.argv or SILENT
PRINT_ALL_CALLERS  = '--print-all-callers' in sys.argv
USE_ASSERT         = not NO_ASSERTS
NOT_QUIET          = not QUIET
PRINT_INJECT_ORDER = (VERYVERBOSE or '--print-inject-order' in sys.argv or
                      '--verbinject' in sys.argv or
                      '--print-imports' in sys.argv or
                      '--verb-inject' in sys.argv or
                      '--verbimport' in sys.argv or
                      '--verbimp' in sys.argv or
                      '--verb-import' in sys.argv or
                      '--verb-import' in sys.argv or
                      '--verbose-import' in sys.argv)
LOGGING_VERBOSE    = VERYVERBOSE or '--verb-logging' in sys.argv

if PRINT_INJECT_ORDER:
    # HACK
    from utool._internal import meta_util_dbg
    from six.moves import builtins
    N = 0
    callername = meta_util_dbg.get_caller_name(N=2 + N, strict=False)
    lineno = meta_util_dbg.get_caller_lineno(N=2 + N, strict=False)
    fmtdict = dict(N=N, lineno=lineno, callername=callername, modname=__name__)
    msg = '[util_inject] N={N} {modname} is imported by {callername} at lineno={lineno}'.format(**fmtdict)
    builtins.print(msg)


def _try_cast(val, type_):
    if type_ in [types.BooleanType] and isinstance(val, six.string_types):
        if val.lower() == 'true':
            newval = True
        elif val.lower() == 'false':
            newval = True
    else:
        newval = type_(val)
    return newval


def get_argflag(flag):
    return flag in sys.argv


# FIXME: on a rrrr this get_argval is taken
# instead of the correct one from util_arg
def get_argval(argstr, type_=None, default=None):
    r"""
    Returns a value of an argument specified on the command line after some flag

    CommandLine:
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="holy grail"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="42"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest=42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest', float)]])" --quest 42

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import sys
        >>> sys.argv.extend(['--spam', 'eggs', '--quest=holy grail', '--ans=42'])
        >>> get_argval('--spam', type_=str, default=None)
        eggs
        >>> get_argval('--quest', type_=str, default=None)
        holy grail
        >>> get_argval('--ans', type_=int, default=None)
        42
    """
    arg_after = default
    if type_ is bool:
        arg_after = False if default is None else default
    try:
        # New for loop way (accounts for =)
        for argx, item in enumerate(sys.argv):
            if item == argstr:
                if argx < len(sys.argv):
                    if type_ is bool:
                        arg_after = True
                    else:
                        arg_after = _try_cast(sys.argv[argx + 1], type_)
            if item.startswith(argstr + '='):
                val_after = ''.join(item.split('=')[1:])
                arg_after = _try_cast(val_after, type_)
    except Exception:
        pass
    return arg_after
