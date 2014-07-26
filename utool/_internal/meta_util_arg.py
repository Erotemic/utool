from __future__ import absolute_import, division, print_function
import types
import sys


def get_arg(arg, type_=None, default=None, **kwargs):
    arg_after = default
    if type_ is bool:
        arg_after = False if default is None else default
    try:
        argx = sys.argv.index(arg)
        if argx < len(sys.argv):
            if type_ is bool:
                arg_after = True
            else:
                try:
                    arg_after_ = sys.argv[argx + 1]
                    if type_ in [types.BooleanType] and isinstance(arg_after_, (str, unicode)):
                        if arg_after_.lower() == 'true':
                            arg_after_ = True
                        elif arg_after_.lower() == 'false':
                            arg_after_ = True
                    else:
                        arg_after_ = type_(arg_after_)
                except Exception:
                    raise
                    pass
                arg_after = arg_after_
    except Exception:
        pass
    return arg_after
