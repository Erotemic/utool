from __future__ import absolute_import, division, print_function
import sys
import six
# Python
import os
import re
#import six
import argparse
from utool import util_inject
from utool import util_type
#from utool import util_print
from utool._internal import meta_util_six, meta_util_arg, meta_util_iter
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[arg]')

#STRICT = '--nostrict' not in sys.argv
DEBUG2       = meta_util_arg.DEBUG2
NO_ASSERTS   = meta_util_arg.NO_ASSERTS
SAFE         = meta_util_arg.SAFE
STRICT       = meta_util_arg.STRICT
REPORT       = meta_util_arg.REPORT
SUPER_STRICT = meta_util_arg.SUPER_STRICT
TRACE        = meta_util_arg.TRACE
USE_ASSERT   = meta_util_arg.USE_ASSERT
SILENT       = meta_util_arg.SILENT
VERBOSE      = meta_util_arg.VERBOSE
VERYVERBOSE  = meta_util_arg.VERYVERBOSE
NOT_QUIET    = meta_util_arg.NOT_QUIET
QUIET        = meta_util_arg.QUIET


#(switch, type, default, help)
__REGISTERED_ARGS__ = []


@profile
def get_argflag(argstr_, default=False, help_='', return_was_specified=False, **kwargs):
    """ Checks if the commandline has a flag or a corresponding noflag """
    global __REGISTERED_ARGS__
    assert isinstance(default, bool), 'default must be boolean'
    argstr_list = meta_util_iter.ensure_iterable(argstr_)
    # arg registration
    __REGISTERED_ARGS__.append((argstr_list, bool, default, help_))
    parsed_val = default
    was_specified = False
    for argstr in argstr_list:
        if not (argstr.find('--') == 0 or (argstr.find('-') == 0 and len(argstr) == 2)):
            raise AssertionError('Invalid argstr: %r' % (argstr,))
        #if argstr.find('--no') == 0:
            #argstr = argstr.replace('--no', '--')
        noarg = argstr.replace('--', '--no')
        if argstr in sys.argv:
            parsed_val = True
            was_specified = True
            break
        elif noarg in sys.argv:
            parsed_val = False
            was_specified = True
            break
    if return_was_specified:
        return parsed_val, was_specified
    else:
        return parsed_val


# TODO: rectify with meta_util_arg
# This has diverged and is now better
#from utool._internal.meta_util_arg import get_argval
@profile
def get_argval(argstr_, type_=None, default=None, help_=None, smartcast=True, return_was_specified=False):
    """ Returns a value of an argument specified on the command line after some flag

    Examples:
        >>> from utool.util_arg import *  # NOQA
        >>> import sys
        >>> sys.argv.extend(['--spam', 'eggs', '--quest=holy grail', '--ans=42'])
        >>> res1 = get_argval('--spam', type_=str, default=None)
        >>> res2 = get_argval('--quest', type_=str, default=None)
        >>> res3 = get_argval('--ans', type_=int, default=None)
        >>> result = ', '.join(map(str, (res1, res2, res3)))
        >>> print(result)
        eggs, holy grail, 42

    CommandLine:
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="holy grail"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="42"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest=42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest', float)]])" --quest 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval(('--nAssign'), int)]])" --nAssign 42
    """
    global __REGISTERED_ARGS__
    #print(argstr_)
    was_specified = False
    arg_after = default
    if type_ is bool:
        arg_after = False if default is None else default
    try:
        # New for loop way (accounts for =)
        argstr_list = meta_util_iter.ensure_iterable(argstr_)
        # arg registration
        __REGISTERED_ARGS__.append((argstr_list, type_, default, help_))

        for argx, item in enumerate(sys.argv):
            for argstr in argstr_list:
                if item == argstr:
                    if type_ is bool:
                        arg_after = True
                        was_specified = True
                        break
                    if argx < len(sys.argv):
                        if type_ is list:
                            # HACK FOR LIST. TODO INTEGRATE
                            arg_after = parse_arglist_hack(argx)
                            if smartcast:
                                arg_after = list(map(util_type.smart_cast2, arg_after))
                        else:
                            arg_after = util_type.try_cast(sys.argv[argx + 1], type_)
                        was_specified = True
                        break
                elif item.startswith(argstr + '='):
                    val_after = ''.join(item.split('=')[1:])
                    if type_ is list:
                        #import utool as ut
                        #ut.embed()
                        # HACK FOR LIST. TODO INTEGRATE
                        val_after_ = val_after.rstrip(']').lstrip('[')
                        arg_after = val_after_.split(',')
                        if smartcast:
                            arg_after = list(map(util_type.smart_cast2, arg_after))
                    else:
                        arg_after = util_type.try_cast(val_after, type_)
                    was_specified = True
                    break
    except Exception as ex:
        import utool as ut
        ut.printex(ex, 'problem in arg_val')
        pass
    if return_was_specified:
        return arg_after, was_specified
    else:
        return arg_after


@profile
def parse_cfgstr_list(cfgstr_list, smartcast=True):
    """
    Parses a list of items in the format
    ['var1:val1', 'var2:val2', 'var3:val3']
    the '=' character can be used instead of the ':' character if desired

    Args:
        cfgstr_list (list):

    Returns:
        dict: cfgdict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> cfgstr_list = ['var1:val1', 'var2:1', 'var3:1.0', 'var4:None']
        >>> smartcast = True
        >>> cfgdict = parse_cfgstr_list(cfgstr_list, smartcast)
        >>> result = str(cfgdict)
        >>> print(result)
        {'var4': None, 'var1': 'val1', 'var3': 1.0, 'var2': 1}
    """
    cfgdict = {}
    for item in cfgstr_list:
        keyval_tup = item.replace('=', ':').split(':')
        assert len(keyval_tup) == 2, '[!] Invalid cfgitem=%r' % (item,)
        key, val = keyval_tup
        if smartcast:
            val = util_type.smart_cast2(val)
        cfgdict[key] = val
    return cfgdict


def parse_arglist_hack(argx):
    arglist = []
    #import utool as ut
    #ut.embed()
    for argx2 in range(argx + 1, len(sys.argv)):
        listarg = sys.argv[argx2]
        if listarg.startswith('-'):
            break
        else:
            arglist.append(listarg)
    return arglist


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


def autogen_argparse2(dpath_list):
    r"""

    FUNCTION IS NOT FULLY IMPLEMENTED CURRENTLY ONLY RETURNS
    LIST OF FLAGS THAT THE PROGRAM SILENTLY TAKES

    Example:
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> dpath_list = [
        ...     ut.truepath('~/code/utool/utool'),
        ...     ut.truepath('~/code/ibeis/ibeis'),
        ...     ut.truepath('~/code/guitool/guitool'),
        ...     ut.truepath('~/code/vtool/vtool'),
        ...     ut.truepath('~/code/plottool/plottool'),
        ... ]
        >>> flagtups_list = autogen_argparse2(dpath_list)
        >>> flagtup_list_ = [ut.regex_replace('[)(\']','',tupstr) for tupstr in ut.flatten(flagtups_list)]
        >>> flagtup_list = ut.flatten([tupstr.split(',') for tupstr in flagtup_list_])
        >>> flagtup_set = set([tupstr.strip() for tupstr in flagtup_list if tupstr.find('=') == -1])
        >>> print('\n'.join(flagtup_set))
    """
    import utool as ut
    import parse
    include_patterns = ['*.py']
    regex_list = ['get_argflag', 'get_argval']
    recursive = True
    result = ut.grep(regex_list, recursive, dpath_list, include_patterns, verbose=True)
    (found_filestr_list, found_lines_list, found_lxs_list) = result
    # TODO: Check to see if in a comment block
    flagtups_list = []
    for found_lines in found_lines_list:
        flagtups = []
        for line in found_lines:
            line_ = ut.regex_replace('#.*', '', line)

            argval_parse_list = [
                '\'{flag}\' in sys.argv',
                'get_argval({flagtup}, type={type}, default={default})',
                'get_argval({flagtup}, {type}, default={default})',
                'get_argval({flagtup}, {type}, {default})',
                'get_argval({flagtup})',
            ]
            argflag_parse_list = [
                'get_argflag({flagtup})',
            ]
            def parse_pattern_list(parse_list, line):
                #result_list = []
                result = None
                for pattern in parse_list:
                    result = parse.parse('{_prefix}' + pattern, line_)
                    if result is not None:
                        break
                        #if len(result_list) > 1:
                        #    print('warning')
                        #result_list.append(result)
                return result
            val_result  = parse_pattern_list(argval_parse_list, line)
            flag_result = parse_pattern_list(argflag_parse_list, line)
            if flag_result is None and val_result is None:
                print('warning1')
            elif flag_result is not None and val_result is not None:
                print('warning2')
            else:
                result = flag_result if val_result is None else val_result
                flagtups.append(result['flagtup'])
        flagtups_list.append(flagtups)
    return flagtups_list


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
    flag = meta_util_six.get_funcname(func)
    if flag.find('no') == 0:
        flag = flag[2:]
    flag = '--' + flag.replace('_', '-')

    def GaurdWrapper(*args, **kwargs):
        # FIXME: the --print-all is a hack
        default_ = kwargs.pop('default', default)
        alias_flags = kwargs.pop('alias_flags', [])
        is_flagged = (get_argflag(flag, default_) or
                      get_argflag('--print-all') or
                      any([get_argflag(_) for _ in alias_flags]))
        if is_flagged:
            indent_lbl = flag.replace('--', '').replace('print-', '')
            print('')
            print('\n+ --- ' + indent_lbl + ' ___')
            #with util_print.Indenter('[%s]' % indent_lbl):
            ret = func(*args, **kwargs)
            print('L ___ ' + indent_lbl + '___\n')
            return ret
        else:
            if not quiet:
                print('\n~~~ %s ~~~' % flag)
    meta_util_six.set_funcname(GaurdWrapper, meta_util_six.get_funcname(func))
    return GaurdWrapper


@profile
def argparse_dict(default_dict_, lbl=None, verbose=VERBOSE,
                  only_specified=False, force_keys={}):
    r"""
    Gets values for a dict based on the command line

    Args:
        default_dict_ (?):
        only_specified (bool): if True only returns keys that are specified on commandline. no defaults.

    Returns:
        dict_: dict_ -  a dictionary

    CommandLine:
        python -m utool.util_arg --test-argparse_dict
        python -m utool.util_arg --test-argparse_dict --flag1
        python -m utool.util_arg --test-argparse_dict --flag2
        python -m utool.util_arg --test-argparse_dict --noflag2
        python -m utool.util_arg --test-argparse_dict --thresh=43
        python -m utool.util_arg --test-argparse_dict --bins=-10

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> default_dict_ = {
        ...    'thresh': -5.333,
        ...    'neg': -5,
        ...    'bins': 8,
        ...    'max': 0.2,
        ...    'flag1': False,
        ...    'flag2': True,
        ... }
        >>> # execute function
        >>> dict_ = argparse_dict(default_dict_)
        >>> # verify results
        >>> result = ut.dict_str(dict_)
        >>> print(result)
    """
    def make_argstrs(key, prefix_list):
        for prefix in prefix_list:
            yield prefix + key
            yield prefix + key.replace('-', '_')
            yield prefix + key.replace('_', '-')

    def get_dictkey_cmdline_val(key, default):
        # see if the user gave a commandline value for this dict key
        type_ = type(default)
        was_specified = False
        if isinstance(default, bool):
            val = default
            if default is True:
                falsekeys = list(set(make_argstrs(key, ['--no', '--no-'])))
                notval, was_specified = get_argflag(falsekeys, return_was_specified=True)
                val = not notval
            elif default is False:
                truekeys = list(set(make_argstrs(key, ['--'])))
                val, was_specified = get_argflag(truekeys, return_was_specified=True)
        else:
            argtup = list(set(make_argstrs(key, ['--'])))
            val, was_specified = get_argval(argtup, type_=type_, default=default, return_was_specified=True)
        return val, was_specified

    dict_  = {}
    for key, default in six.iteritems(default_dict_):
        val, was_specified = get_dictkey_cmdline_val(key, default)
        if not only_specified or was_specified or key in force_keys:
            dict_[key] = val
    #dict_ = {key: get_dictkey_cmdline_val(key, default) for key, default in six.iteritems(default_dict_)}

    if verbose:
        for key in dict_:
            if dict_[key] != default_dict_[key]:
                print('[argparse_dict] GOT ARGUMENT: cfgdict[%r] = %r' % (key, dict_[key]))

    if get_argflag(('--help', '--helpx')):
        import utool as ut
        print('COMMAND LINE IS ACCEPTING THESE PARAMS WITH DEFAULTS:')
        if lbl is not None:
            print(lbl)
        print(ut.align(ut.dict_str(default_dict_, sorted_=True), ':'))
        if get_argflag('--helpx'):
            sys.exit(1)
    return dict_


def get_argv_tail(scriptname):
    """ gets the rest of the arguments after a script has been invoked

    Args:
        scriptname (?):

    CommandLine:
        python -m utool.util_arg --test-get_argv_tail

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> # build test data
        >>> scriptname = 'util_arg.py'
        >>> # execute function
        >>> result = get_argv_tail(scriptname)
        >>> # verify results
        >>> print(result)
    """
    import utool as ut
    modname = ut.get_argval('-m', help_='specify module name to profile')
    #print(sys.argv)
    if modname is not None:
        modpath = ut.get_modpath_from_modname(modname)
        #print(modpath)
        argvx = sys.argv.index(modname) + 1
        argv_tail = [modpath] + sys.argv[argvx:]
    else:
        try:
            argvx = sys.argv.index(scriptname)
        except ValueError:
            for argvx, arg in enumerate(sys.argv):
                if scriptname in arg:
                    break
            #print('sys.argv = %r' % (sys.argv,))
        argv_tail = sys.argv[(argvx + 1):]
    return argv_tail


# alias
parse_dict_from_argv = argparse_dict
get_dict_vals_from_commandline = argparse_dict


if __name__ == '__main__':
    """
    CommandLine:
        python utool/util_arg.py --test-autogen_argparse2
    """
    from utool import util_tests
    util_tests.doctest_funcs([autogen_argparse2])
