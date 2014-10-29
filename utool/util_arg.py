from __future__ import absolute_import, division, print_function
import sys
import six
# Python
import os
import re
#import six
import argparse
from .util_type import try_cast
from .util_inject import inject
from .util_print import Indenter
from utool._internal.meta_util_six import get_funcname, set_funcname
print, print_, printDBG, rrr, profile = inject(__name__, '[arg]')

#STRICT = '--nostrict' not in sys.argv
DEBUG2 = '--debug2' in sys.argv
IGNORE_TRACEBACK = '--smalltb' in sys.argv or '--ignoretb' in sys.argv
NO_ASSERTS = ('--no-assert' in sys.argv)
QUIET = '--quiet' in sys.argv
SAFE = '--safe' in sys.argv
STRICT = '--strict' not in sys.argv
REPORT = '--report' not in sys.argv
SUPER_STRICT = '--super-strict' in sys.argv or '--superstrict' in sys.argv
TRACE = '--trace' in sys.argv
USE_ASSERT = not NO_ASSERTS
VERBOSE = '--verbose' in sys.argv
VERYVERBOSE = '--very-verbose' in sys.argv or '-veryverbose' in sys.argv
NOT_QUIET = not QUIET


# TODO: rectify with meta_util_arg
# This has diverged and is now better
#from ._internal.meta_util_arg import get_argval
def get_argval(argstr_, type_=None, default=None, help_=None):
    """ Returns a value of an argument specified on the command line after some flag


    Examples:
        >>> from utool.util_arg import *  # NOQA
        >>> import sys
        >>> sys.argv.extend(['--spam', 'eggs', '--quest=holy grail', '--ans=42'])
        >>> get_argval('--spam', type_=str, default=None)
        eggs
        >>> get_argval('--quest', type_=str, default=None)
        holy grail
        >>> get_argval('--ans', type_=int, default=None)
        42


        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="holy grail"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="42"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest=42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest', float)]])" --quest 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval(('--nAssign'), int)]])" --nAssign 42
    """
    arg_after = default
    if type_ is bool:
        arg_after = False if default is None else default
    try:
        # New for loop way (accounts for =)
        if isinstance(argstr_, six.string_types):
            argstr_list = (argstr_,)
        else:
            argstr_list = argstr_
        for argx, item in enumerate(sys.argv):
            for argstr in argstr_list:
                if item == argstr:
                    if argx < len(sys.argv):
                        if type_ is bool:
                            arg_after = True
                        else:
                            arg_after = try_cast(sys.argv[argx + 1], type_)
                if item.startswith(argstr + '='):
                    val_after = ''.join(item.split('=')[1:])
                    arg_after = try_cast(val_after, type_)
    except Exception:
        pass
    return arg_after


def get_argflag(arg, default=False, help_='', **kwargs):
    """ Checks if the commandline has a flag or a corresponding noflag """
    if isinstance(arg, (tuple, list)):
        arg_list = arg
    else:
        arg_list = [arg]
    for arg in arg_list:
        if not (arg.find('--') == 0 or (arg.find('-') == 0 and len(arg) == 2)):
            raise AssertionError(arg)
        #if arg.find('--no') == 0:
            #arg = arg.replace('--no', '--')
        noarg = arg.replace('--', '--no')
        if arg in sys.argv:
            return True
        elif noarg in sys.argv:
            return False
    return default


# Backwards Compatibility Aliases
get_arg  = get_argval
get_flag  = get_argflag


#def argv_flag(name, default, **kwargs):
#    if name.find('--') == 0:
#        name = name[2:]
#    if '--' + name in sys.argv and default is False:
#        return True
#    if '--no' + name in sys.argv and default is True:
#        return False
#    return default


# ---- OnTheFly argparse ^^^^
# ---- Documented argparse VVVV


def switch_sanataize(switch):
    if isinstance(switch, six.string_types):
        dest = switch.strip('-').replace('-', '_')
    else:
        if isinstance(switch, tuple):
            switch = switch
        elif isinstance(switch, list):
            switch = tuple(switch)
        dest = switch[0].strip('-').replace('-', '_')
    return dest, switch


def fuzzy_int(str_):
    """
    lets some special strings be interpreted as ints
    """
    try:
        ret = int(str_)
        return ret
    except Exception:
        if re.match(r'\d*:\d*:?\d*', str_):
            return tuple(range(*map(int, str_.split(':'))))
        raise


class ArgumentParser2(object):
    """ Wrapper around argparse.ArgumentParser with convinence functions """
    def __init__(self, parser):
        self.parser = parser

    def add_arg(self, switch, *args, **kwargs):
        #print('[argparse2] add_arg(%r) ' % (switch,))
        if isinstance(switch, tuple):
            args = tuple(list(switch) + list(args))
            return self.parser.add_argument(*args, **kwargs)
        else:
            return self.parser.add_argument(switch, *args, **kwargs)

    def add_meta(self, switch, type, default=None, help='', **kwargs):
        #print('[argparse2] add_meta()')
        dest, switch = switch_sanataize(switch)
        self.add_arg(switch, metavar=dest, type=type, default=default, help=help, **kwargs)

    def add_flag(self, switch, default=False, **kwargs):
        #print('[argparse2] add_flag()')
        action = 'store_false' if default else 'store_true'
        dest, switch = switch_sanataize(switch)
        self.add_arg(switch, dest=dest, action=action, default=default, **kwargs)

    def add_int(self, switch, *args, **kwargs):
        self.add_meta(switch, fuzzy_int,  *args, **kwargs)

    def add_intlist(self, switch, *args, **kwargs):
        self.add_meta(switch, fuzzy_int,  *args, nargs='*', **kwargs)

    add_ints = add_intlist

    def add_strlist(self, switch, *args, **kwargs):
        self.add_meta(switch, str,  *args, nargs='*', **kwargs)

    add_strs = add_strlist

    def add_float(self, switch, *args, **kwargs):
        self.add_meta(switch, float, *args, **kwargs)

    def add_str(self, switch, *args, **kwargs):
        self.add_meta(switch, str, *args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        return ArgumentParser2(self.parser.add_argument_group(*args, **kwargs))


def make_argparse2(prog='Program', description='', *args, **kwargs):
    formatter_classes = [
        argparse.RawDescriptionHelpFormatter,
        argparse.RawTextHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter]
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description,
                                     prefix_chars='+-',
                                     formatter_class=formatter_classes[2], *args,
                                     **kwargs)
    return ArgumentParser2(parser)


# Decorators which control program flow based on sys.argv
# the decorated function does not execute without its corresponding
# flag

def get_fpath_args(arglist_=None, pat='*'):
    import utool
    if arglist_ is None:
        arglist_ = sys.argv[1:]
    input_path_list = []
    for input_path in arglist_:
        input_path = utool.truepath(input_path)
        if os.path.isdir(input_path):
            input_path_list.extend(utool.glob(input_path, pat, recursive=False, with_dirs=False))
        else:
            input_path_list.append(input_path)
    return input_path_list


def argv_flag_dec(func):
    """
    Decorators which control program flow based on sys.argv
    the decorated function does not execute without its corresponding
    flag
    """

    return __argv_flag_dec(func, default=False)


def argv_flag_dec_true(func):
    return __argv_flag_dec(func, default=True)


def __argv_flag_dec(func, default=False, quiet=QUIET):
    flag = get_funcname(func)
    if flag.find('no') == 0:
        flag = flag[2:]
    flag = '--' + flag.replace('_', '-')

    def GaurdWrapper(*args, **kwargs):
        # FIXME: the --print-all is a hack
        if get_argflag(flag, default) or get_argflag('--print-all'):
            indent_lbl = flag.replace('--', '').replace('print-', '')
            print('')
            print('\n+++ ' + indent_lbl + ' +++')
            with Indenter('[%s]' % indent_lbl):
                return func(*args, **kwargs)
            print('')
        else:
            if not quiet:
                print('\n~~~ %s ~~~' % flag)
    set_funcname(GaurdWrapper, get_funcname(func))
    return GaurdWrapper
