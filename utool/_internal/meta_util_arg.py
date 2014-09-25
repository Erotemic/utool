from __future__ import absolute_import, division, print_function
import types
import six
import sys


#def get_arg(arg, type_=None, default=None):
#    arg_after = default
#    if type_ is bool:
#        arg_after = False if default is None else default
#    try:
#        argx = sys.argv.index(arg)
#        if argx < len(sys.argv):
#            if type_ is bool:
#                arg_after = True
#            else:
#                try:
#                    arg_after_ = sys.argv[argx + 1]
#                    if type_ in [types.BooleanType] and isinstance(arg_after_, six.string_types):
#                        if arg_after_.lower() == 'true':
#                            arg_after_ = True
#                        elif arg_after_.lower() == 'false':
#                            arg_after_ = True
#                    else:
#                        arg_after_ = type_(arg_after_)
#                except Exception:
#                    raise
#                    pass
#                arg_after = arg_after_
#    except Exception:
#        pass
#    return arg_after


def _try_cast(val, type_):
    if type_ in [types.BooleanType] and isinstance(val, six.string_types):
        if val.lower() == 'true':
            newval = True
        elif val.lower() == 'false':
            newval = True
    else:
        newval = type_(val)
    return newval


def get_arg(argstr, type_=None, default=None):
    """ Returns a value of an argument specified on the command line after some flag

    python -c "import utool; print([(type(x), x) for x in [utool.get_arg('--quest')]])" --quest="holy grail"
    python -c "import utool; print([(type(x), x) for x in [utool.get_arg('--quest')]])" --quest="42"
    python -c "import utool; print([(type(x), x) for x in [utool.get_arg('--quest')]])" --quest=42
    python -c "import utool; print([(type(x), x) for x in [utool.get_arg('--quest')]])" --quest 42
    python -c "import utool; print([(type(x), x) for x in [utool.get_arg('--quest', float)]])" --quest 42

    >>> from utool.util_arg import *  # NOQA
    >>> import sys
    >>> sys.argv.extend(['--spam', 'eggs', '--quest=holy grail', '--ans=42'])
    >>> get_arg('--spam', type_=str, default=None)
    eggs
    >>> get_arg('--quest', type_=str, default=None)
    holy grail
    >>> get_arg('--ans', type_=int, default=None)
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
