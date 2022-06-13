# -*- coding: utf-8 -*-
"""
Handles command line parsing
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import six
import os
from six.moves import builtins
import itertools
import argparse
from utool import util_inject
from utool import util_type
from utool._internal import meta_util_six, meta_util_arg, meta_util_iter
print, rrr, profile = util_inject.inject2(__name__)

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

DEBUG = 0


#(switch, type, default, help)
# TODO: make a static help file available via printing
__REGISTERED_ARGS__ = []


def aug_sysargv(cmdstr):
    """ DEBUG FUNC modify argv to look like you ran a command """
    import shlex
    argv = shlex.split(cmdstr)
    sys.argv.extend(argv)
    #return argv


def get_module_verbosity_flags(*labels):
    """ checks for standard flags for enableing module specific verbosity """
    verbose_prefix_list = ['--verbose-', '--verb', '--verb-']
    veryverbose_prefix_list = ['--veryverbose-', '--veryverb', '--veryverb-']
    verbose_flags = tuple(
        [prefix + lbl for prefix, lbl in
         itertools.product(verbose_prefix_list, labels)])
    veryverbose_flags = tuple(
        [prefix + lbl for prefix, lbl in
         itertools.product(veryverbose_prefix_list, labels)])
    veryverbose_module = get_argflag(veryverbose_flags) or VERYVERBOSE
    verbose_module = (get_argflag(verbose_flags) or veryverbose_module or VERBOSE)
    if veryverbose_module:
        verbose_module = 2
    return verbose_module, veryverbose_module

get_verbflag = get_module_verbosity_flags


def reset_argrecord():
    """ forgets about the args already parsed """
    global __REGISTERED_ARGS__
    __REGISTERED_ARGS__ = []


def _register_arg(argstr_list, type_, default, help_):
    # TODO REGISTER PARENTS
    global __REGISTERED_ARGS__
    __REGISTERED_ARGS__.append((argstr_list, type_, default, help_))


def autogen_argparse_block(extra_args=[]):
    """
    SHOULD TURN ANY REGISTERED ARGS INTO A A NEW PARSING CONFIG
    FILE FOR BETTER --help COMMANDS

    import utool as ut
    __REGISTERED_ARGS__ = ut.util_arg.__REGISTERED_ARGS__

    Args:
        extra_args (list): (default = [])

    CommandLine:
        python -m utool.util_arg --test-autogen_argparse_block

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> extra_args = []
        >>> result = ut.autogen_argparse_block(extra_args)
        >>> print(result)
    """
    #import utool as ut  # NOQA
    #__REGISTERED_ARGS__
    # TODO FINISHME

    grouped_args = []
    # Group similar a args
    for argtup in __REGISTERED_ARGS__:
        argstr_list, type_, default, help_ = argtup
        argstr_set = set(argstr_list)
        # <MULTIKEY_SETATTR>
        # hack in multikey setattr n**2 yuck
        found = False
        for index, (keyset, vals) in enumerate(grouped_args):
            if len(keyset.intersection(argstr_set)) > 0:
                # update
                keyset.update(argstr_set)
                vals.append(argtup)
                found = True
                break
        if not found:
            new_keyset = argstr_set
            new_vals = [argtup]
            grouped_args.append((new_keyset, new_vals))
        # </MULTIKEY_SETATTR>
    # DEBUG
    multi_groups = []
    for keyset, vals in grouped_args:
        if len(vals) > 1:
            multi_groups.append(vals)
    if len(multi_groups) > 0:
        import utool as ut
        print('Following arg was specified multiple times')
        print(ut.repr4(multi_groups, newlines=2))


def get_argflag(argstr_, default=False, help_='', return_specified=None,
                need_prefix=True, return_was_specified=False, argv=None,
                debug=None,
                **kwargs):
    """
    Checks if the commandline has a flag or a corresponding noflag

    Args:
        argstr_ (str, list, or tuple): the flag to look for
        default (bool): dont use this (default = False)
        help_ (str): a help string (default = '')
        return_specified (bool): returns if flag was specified or not (default = False)

    Returns:
        tuple: (parsed_val, was_specified)

    TODO:
        depricate return_was_specified

    CommandLine:
        python -m utool.util_arg --exec-get_argflag --noface --exec-mode
        python -m utool.util_arg --exec-get_argflag --foo --exec-mode
        python -m utool.util_arg --exec-get_argflag --no-foo --exec-mode
        python -m utool.util_arg --exec-get_argflag --foo=True --exec-mode
        python -m utool.util_arg --exec-get_argflag --foo=False  --exec-mode

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> argstr_ = '--foo'
        >>> default = False
        >>> help_ = ''
        >>> return_specified = True
        >>> (parsed_val, was_specified) = get_argflag(argstr_, default, help_, return_specified)
        >>> result = ('(parsed_val, was_specified) = %s' % (str((parsed_val, was_specified)),))
        >>> print(result)
    """
    if argv is None:
        argv = sys.argv
    assert isinstance(default, bool), 'default must be boolean'
    argstr_list = meta_util_iter.ensure_iterable(argstr_)
    #if VERYVERBOSE:
    #    print('[util_arg] checking argstr_list=%r' % (argstr_list,))
    # arg registration
    _register_arg(argstr_list, bool, default, help_)
    parsed_val = default
    was_specified = False

    if debug is None:
        debug = DEBUG

    # Check environment variables for default as well as argv
    import os
    #"""
    #set UTOOL_NOCNN=True
    #export UTOOL_NOCNN True
    #"""
    #argv_orig = argv[:]
    # HACK: make this not happen very time you loop
    for key, val in os.environ.items():
        key = key.upper()
        sentinal = 'UTOOL_'
        if key.startswith(sentinal):
            flag = '--' + key[len(sentinal):].lower().replace('_', '-')
            if val.upper() in ['TRUE', 'ON']:
                pass
            elif val.upper() in ['FALSE', 'OFF']:
                continue
            else:
                continue
                #flag += '=False'
            new_argv = [flag]
            argv = argv[:] + new_argv
            if debug:
                print('ENV SPECIFIED COMMAND LINE')
                print('argv.extend(new_argv=%r)' % (new_argv,))

    for argstr in argstr_list:
        #if VERYVERBOSE:
        #    print('[util_arg]   * checking argstr=%r' % (argstr,))
        if not (argstr.find('--') == 0 or (argstr.find('-') == 0 and len(argstr) == 2)):
            raise AssertionError('Invalid argstr: %r' % (argstr,))
        if not need_prefix:
            noprefix = argstr.replace('--', '')
            if noprefix in argv:
                parsed_val = True
                was_specified = True
                break
        #if argstr.find('--no') == 0:
            #argstr = argstr.replace('--no', '--')
        noarg = argstr.replace('--', '--no')
        if argstr in argv:
            parsed_val = True
            was_specified = True
            #if VERYVERBOSE:
            #    print('[util_arg]   * ...WAS_SPECIFIED. AND PARSED')
            break
        elif noarg in argv:
            parsed_val = False
            was_specified = True
            #if VERYVERBOSE:
            #    print('[util_arg]   * ...WAS_SPECIFIED. AND NOT PARSED')
            break
        elif argstr + '=True' in argv:
            parsed_val = True
            was_specified = True
            break
        elif argstr + '=False' in argv:
            parsed_val = False
            was_specified = True
            break

    if return_specified is None:
        return_specified = return_was_specified

    if return_specified:
        return parsed_val, was_specified
    else:
        return parsed_val


# TODO: rectify with meta_util_arg
# This has diverged and is now better
#from utool._internal.meta_util_arg import get_argval
#@profile
def get_argval(argstr_, type_=None, default=None, help_=None, smartcast=True,
               return_specified=None, argv=None, verbose=None,
               debug=None, return_was_specified=False, pos=None):
    r"""
    Returns a value of an argument specified on the command line after some flag

    Args:
        argstr_ (str or tuple): string or tuple of strings denoting the command line values to parse
        type_ (None): type of the variable to parse (default = None)
        default (None): (default = None)
        help_ (None): help for this argument (not fully integrated) (default = None)
        smartcast (bool): tries to be smart about casting the parsed strings (default = True)
        return_specified (bool): (default = False)
        argv (None): override sys.argv with custom command line vector (default = None)
        pos (int): if specified the argument can also be found in position `pos` of the command line varargs

    TODO:
        depricate return_was_specified

    CommandLine:
        python -m utool.util_arg --test-get_argval
        python -m utool.util_arg --exec-get_argval:0
        python -m utool.util_arg --exec-get_argval:1
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="holy grail"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest="42"
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest=42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest')]])" --quest 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval('--quest', float)]])" --quest 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval(('--nAssign'), int)]])" --nAssign 42
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval(('--test'), str)]])" --test
        python -c "import utool; print([(type(x), x) for x in [utool.get_argval(('--test'), str)]])" --test "foobar is good" --youbar ok

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> import sys
        >>> argv = ['--spam', 'eggs', '--quest=holy grail', '--ans=42', '--the-val=1,2,3']
        >>> # specify a list of args and kwargs to get_argval
        >>> argstr_kwargs_list = [
        >>>     ('--spam',                    dict(type_=str, default=None, argv=argv)),
        >>>     ('--quest',                   dict(type_=str, default=None, argv=argv)),
        >>>     (('--ans', '--foo'),          dict(type_=int, default=None, argv=argv)),
        >>>     (('--not-there', '--absent'), dict(argv=argv)),
        >>>     ('--the_val',                 dict(type_=list, argv=argv)),
        >>>     ('--the-val',                 dict(type_=list, argv=argv)),
        >>> ]
        >>> # Execute the command with for each of the test cases
        >>> res_list = []
        >>> argstr_list = ut.get_list_column(argstr_kwargs_list, 0)
        >>> for argstr_, kwargs in argstr_kwargs_list:
        >>>     res = get_argval(argstr_, **kwargs)
        >>>     res_list.append(res)
        >>> result = ut.repr2(ut.odict(zip(argstr_list, res_list)), nl=1)
        >>> result = result.replace('u\'', '\'')  # hack
        >>> print(result)
        {
            '--spam': 'eggs',
            '--quest': 'holy grail',
            ('--ans', '--foo'): 42,
            ('--not-there', '--absent'): None,
            '--the_val': [1, 2, 3],
            '--the-val': [1, 2, 3],
        }

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> import sys
        >>> argv = ['--slice1', '::', '--slice2=4:', '--slice3=::4', '--slice4', '[1,2,3,4]', '--slice5=3']
        >>> # specify a list of args and kwargs to get_argval
        >>> argstr_kwargs_list = [
        >>>     ('--slice1',            dict(type_='fuzzy_subset', default=None, argv=argv)),
        >>>     ('--slice2',            dict(type_='fuzzy_subset', default=None, argv=argv)),
        >>>     ('--slice3',            dict(type_='fuzzy_subset', default=None, argv=argv)),
        >>>     ('--slice4',            dict(type_='fuzzy_subset', default=None, argv=argv)),
        >>>     ('--slice5',            dict(type_='fuzzy_subset', default=None, argv=argv)),
        >>> ]
        >>> # Execute the command with for each of the test cases
        >>> res_list = []
        >>> argstr_list = ut.get_list_column(argstr_kwargs_list, 0)
        >>> list1 = [1, 3, 5, 7, 9]
        >>> import numpy as np
        >>> list2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 1]])
        >>> for argstr_, kwargs in argstr_kwargs_list:
        >>>     res = get_argval(argstr_, **kwargs)
        >>>     print('---')
        >>>     print('res = %r' % (res,))
        >>>     print('list1[%r=%r] = %r' % (argstr_, res, ut.take(list1, res),))
        >>>     print('list2[%r=%r] = %r' % (argstr_, res, list2[res].tolist(),))
        >>>     res_list.append(res)
        >>> result = ut.repr4(ut.odict(zip(argstr_list, res_list)))
        >>> result = result.replace('u\'', '\'')  # hack
        >>> print(result)

    """
    if verbose is None:
        pass
        # verbose = VERYVERBOSE

    if debug is None:
        debug = DEBUG
        # debug = VERYVERBOSE

    if argv is None:
        argv = sys.argv

    #verbose = 1

    if verbose:
        print('[get_argval] Searching Commandline for argstr_=%r' % (argstr_,))
        #print('[get_argval]  * type_ = %r' % (type_,))
        #print('[get_argval]  * default = %r' % (default,))
        #print('[get_argval]  * help_ = %r' % (help_,))
        #print('[get_argval]  * smartcast = %r' % (smartcast,))

    if return_specified is None:
        return_specified = return_was_specified

    #print(argstr_)
    was_specified = False
    arg_after = default
    if type_ is bool:
        arg_after = False if default is None else default
    try:
        # New for loop way (accounts for =)
        argstr_list = meta_util_iter.ensure_iterable(argstr_)
        # arg registration
        _register_arg(argstr_list, type_, default, help_)

        # expand out hypens
        EXPAND_HYPENS = True
        if EXPAND_HYPENS:
            argstr_list2 = []
            seen_ = set([])
            for argstr in argstr_list:
                if argstr not in seen_:
                    argstr_list2.append(argstr)
                    seen_.add(argstr)
                if argstr.startswith('--'):
                    num = 2
                elif argstr.startswith('-'):
                    num = 1
                else:
                    continue
                argstr2_0 = argstr[0:num] + argstr[num:].replace('_', '-')
                argstr2_1 = argstr[0:num] + argstr[num:].replace('-', '_')
                if argstr2_0 not  in seen_:
                    argstr_list2.append(argstr2_0)
                    seen_.add(argstr2_0)
                if argstr2_1 not  in seen_:
                    argstr_list2.append(argstr2_1)
                    seen_.add(argstr2_1)
            argstr_list = argstr_list2

        # Check environment variables for default as well as argv
        import os
        """
        set UTOOL_NOCNN=True
        export UTOOL_NOCNN True
        """
        #argv_orig = argv[:]
        for key, val in os.environ.items():
            key = key.upper()
            sentinal = 'UTOOL_'
            if key.startswith(sentinal):
                key = '--' + key[len(sentinal):].lower()
                new_argv = [key, val]
                if val.upper() in ['TRUE', 'FALSE', 'ON', 'OFF']:
                    # handled by get_argflag
                    continue
                argv = argv[:] + new_argv
                if debug:
                    print('argv.extend(new_argv=%r)' % (new_argv,))

        for argx, item in enumerate(argv):
            for argstr in argstr_list:
                if item == argstr:
                    if type_ is bool:
                        if debug:
                            print('[get_argval] ... argstr=%r' % (argstr,))
                            print('[get_argval] ... Found bool argx=%r' % (argx,))
                        arg_after = True
                        was_specified = True
                        break
                    if argx < len(argv):
                        if type_ is list:
                            # HACK FOR LIST. TODO INTEGRATE
                            if debug:
                                print('[get_argval] ... argstr=%r' % (argstr,))
                                print('[get_argval] ... Found noequal list argx=%r' % (argx,))
                            arg_after = parse_arglist_hack(argx, argv=argv)
                            if debug:
                                print('[get_argval] ... arg_after=%r' % (arg_after,))
                                print('argv=%r' % (argv,))
                            if smartcast:
                                arg_after = list(map(util_type.smart_cast2, arg_after))
                                if debug:
                                    print('[get_argval] ... smartcast arg_after=%r' % (arg_after,))
                        else:
                            if debug:
                                print('[get_argval] ... argstr=%r' % (argstr,))
                                print('[get_argval] ... Found type_=%r argx=%r' % (type_, argx,))
                            arg_after = argv[argx + 1]
                            if type_ is not None:
                                arg_after = util_type.try_cast(arg_after, type_)
                            elif smartcast:
                                arg_after = util_type.smart_cast2(arg_after)
                        if was_specified:
                            print('WARNING: argstr=%r already specified' % (argstr,))
                        was_specified = True
                        break
                elif item.startswith(argstr + '='):
                    val_after = ''.join(item.split('=')[1:])
                    if type_ is list:
                        # HACK FOR LIST. TODO INTEGRATE
                        if verbose:
                            print('[get_argval] ... Found equal list')
                        val_after_ = val_after.rstrip(']').lstrip('[')
                        if True:
                            # Hacker way to be less hacky about parsing lists
                            from utool import util_gridsearch
                            blocks = util_gridsearch.parse_nestings(val_after_)
                            sentinal = '##COM&&'
                            changed = [(block[0], block[1].replace(',', sentinal))
                                       if block[0] == 'nonNested' else block
                                       for block in blocks]
                            val_after2 = util_gridsearch.recombine_nestings(changed)
                            arg_after = val_after2.split(sentinal)
                        else:
                            arg_after = val_after_.split(',')
                        if smartcast:
                            arg_after = list(map(util_type.smart_cast2, arg_after))
                    else:
                        if type_ is None:
                            if smartcast:
                                arg_after = util_type.smart_cast2(val_after)
                            else:
                                arg_after = val_after
                        else:
                            arg_after = util_type.try_cast(val_after, type_)
                            if not isinstance(type_, six.string_types) and issubclass(type_, six.string_types):
                                if arg_after == 'None':
                                    # hack
                                    arg_after = None
                    if was_specified:
                        print('WARNING: argstr=%r already specified' % (argstr,))
                    was_specified = True
                    break
    except Exception as ex:
        import utool as ut
        ut.printex(ex, 'problem in arg_val', keys=['type_'])
        if ut.SUPER_STRICT:
            raise
        pass
    if not was_specified and pos is not None:
        varargs = get_cmdline_varargs(argv)
        if len(varargs) > pos:
            arg_after = varargs[pos]
            assert type_ is not list, 'list not handled yet'
            if type_ is not None:
                arg_after = util_type.try_cast(arg_after, type_)
            elif smartcast:
                arg_after = util_type.smart_cast2(arg_after)

    if verbose:
        print('[get_argval] ... Parsed arg_after=%r, was_specified=%r' % (arg_after, was_specified))
    if return_specified:
        return arg_after, was_specified
    else:
        return arg_after


@profile
def parse_cfgstr_list(cfgstr_list, smartcast=True, oldmode=True):
    r"""
    Parses a list of items in the format
    ['var1:val1', 'var2:val2', 'var3:val3']
    the '=' character can be used instead of the ':' character if desired

    TODO: see ut.parse_cfgstr3

    Args:
        cfgstr_list (list):

    Returns:
        dict: cfgdict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> cfgstr_list = ['var1=val1', 'var2=1', 'var3=1.0', 'var4=None', 'var5=[1,2,3]', 'var6=(a,b,c)']
        >>> smartcast = True
        >>> cfgdict = parse_cfgstr_list(cfgstr_list, smartcast, oldmode=False)
        >>> result = ut.repr2(cfgdict, sorted_=True, newlines=False)
        >>> print(result)
        {'var1': 'val1', 'var2': 1, 'var3': 1.0, 'var4': None, 'var5': [1, 2, 3], 'var6': ('a', 'b', 'c')}

        {'var1': 'val1', 'var2': 1, 'var3': 1.0, 'var4': None}

        {'var4': None, 'var1': 'val1', 'var3': 1.0, 'var2': 1}
    """
    cfgdict = {}
    for item in cfgstr_list:
        if item == '':
            continue
        if oldmode:
            keyval_tup = item.replace('=', ':').split(':')
            assert len(keyval_tup) == 2, '[!] Invalid cfgitem=%r' % (item,)
            key, val = keyval_tup
        else:
            keyval_tup = item.split('=')
            if len(keyval_tup) == 1:
                # single specifications are interpeted as booleans
                key = keyval_tup[0]
                val = True
            else:
                assert len(keyval_tup) >= 2, '[!] Invalid cfgitem=%r' % (item,)
                key, val = keyval_tup[0], '='.join(keyval_tup[1:])
        if smartcast:
            val = util_type.smart_cast2(val)
        cfgdict[key] = val
    return cfgdict


def parse_arglist_hack(argx, argv=None):
    if argv is None:
        argv = sys.argv
    arglist = []
    #import utool as ut
    #ut.embed()
    for argx2 in range(argx + 1, len(argv)):
        listarg = argv[argx2]
        if listarg.startswith('-'):
            break
        else:
            arglist.append(listarg)
    return arglist


def get_arg_dict(argv=None, prefix_list=['--'], type_hints={}):
    r"""
    Yet another way for parsing args

    CommandLine:
        python -m utool.util_arg --exec-get_arg_dict
        python -m utool.util_arg --test-get_arg_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> import shlex
        >>> argv = shlex.split('--test-show_name --name=IBEIS_PZ_0303 --db testdb3 --save "~/latex/crall-candidacy-2015/figures/IBEIS_PZ_0303.jpg" --dpath figures --caption="Shadowed"  --figsize=11,3 --no-figtitle -t foo bar baz biz --notitle')
        >>> arg_dict = ut.get_arg_dict(argv, prefix_list=['--', '-'], type_hints={'t': list})
        >>> result = ut.repr2(arg_dict, nl=1)
        >>> # verify results
        >>> print(result)
        {
            'caption': 'Shadowed',
            'db': 'testdb3',
            'dpath': 'figures',
            'figsize': '11,3',
            'name': 'IBEIS_PZ_0303',
            'no-figtitle': True,
            'notitle': True,
            'save': '~/latex/crall-candidacy-2015/figures/IBEIS_PZ_0303.jpg',
            't': ['foo', 'bar', 'baz', 'biz'],
            'test-show_name': True,
        }
    """
    if argv is None:
        argv = sys.argv
    arg_dict = {}

    def startswith_prefix(arg):
        return any([arg.startswith(prefix) for prefix in prefix_list])

    def argx_has_value(argv, argx):
        # Check if has a value
        if argv[argx].find('=') > -1:
            return True
        if argx + 1 < len(argv) and not startswith_prefix(argv[argx + 1]):
            return True
        return False

    def get_arg_value(argv, argx, argname):
        if argv[argx].find('=') > -1:
            return '='.join(argv[argx].split('=')[1:])
        else:
            type_ = type_hints.get(argname, None)
            if type_ is None:
                return argv[argx + 1]
            else:
                return parse_arglist_hack(argx, argv=argv)

    for argx in range(len(argv)):
        arg = argv[argx]
        for prefix in prefix_list:
            if arg.startswith(prefix):
                argname = arg[len(prefix):]
                if argx_has_value(argv, argx):
                    if arg.find('=') > -1:
                        argname = arg[len(prefix):arg.find('=')]
                    argvalue = get_arg_value(argv, argx, argname)
                    arg_dict[argname] = argvalue
                else:
                    arg_dict[argname] = True
                break
    return arg_dict

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
        self.add_meta(switch, util_type.fuzzy_int,  *args, **kwargs)

    def add_intlist(self, switch, *args, **kwargs):
        self.add_meta(switch, util_type.fuzzy_int,  *args, nargs='*', **kwargs)

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
        >>> # DISABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> dpath_list = [
        ...     ut.truepath('~/code/utool/utool'),
        ...     ut.truepath('~/code/ibeis/ibeis'),
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


def argv_flag_dec(*argin, **kwargs):
    """
    Decorators which control program flow based on sys.argv
    the decorated function does not execute without its corresponding
    flag

    Kwargs:
        default, quiet, indent, default

    ReturnKwargs:
        alias_flags
    """
    kwargs = kwargs.copy()
    kwargs['default'] = kwargs.get('default', False)
    from utool import util_decor
    @util_decor.ignores_exc_tb(outer_wrapper=False)
    def wrap_argv_flag_dec(func):
        return __argv_flag_dec(func, **kwargs)

    assert len(argin) < 2, 'specify 0 or 1 args'

    if len(argin) == 1 and util_type.is_funclike(argin[0]):
        func = argin[0]
        return wrap_argv_flag_dec(func)
    else:
        return wrap_argv_flag_dec


def argv_flag_dec_true(func, **kwargs):
    return __argv_flag_dec(func, default=True, **kwargs)


def __argv_flag_dec(func, default=False, quiet=QUIET, indent=False):
    """
    Logic for controlling if a function gets called based on command line
    """
    from utool import util_decor
    flagname = meta_util_six.get_funcname(func)
    if flagname.find('no') == 0:
        flagname = flagname[2:]

    flags = (
        '--' + flagname.replace('_', '-'),
        '--' + flagname,
    )

    @util_decor.ignores_exc_tb(outer_wrapper=False)
    def GaurdWrapper(*args, **kwargs):
        from utool import util_print
        # FIXME: the --print-all is a hack
        default_ = kwargs.pop('default', default)
        alias_flags = kwargs.pop('alias_flags', [])
        is_flagged = (get_argflag(flags, default_) or
                      get_argflag('--print-all') or
                      any([get_argflag(_) for _ in alias_flags]))
        if flagname in kwargs:
            is_flagged = kwargs.pop(flagname)
        if is_flagged:
            func_label = flags[0].replace('--', '').replace('print-', '')
            # print('')
            print('\n+ --- ' + func_label + ' ___')
            use_indent = indent is not False
            if indent is True:
                indent_ = '[%s]' % func_label
            else:
                indent_ = indent
            with util_print.Indenter(indent_, enabled=use_indent):
                ret = func(*args, **kwargs)
            print('L ___ ' + func_label + '___\n')
            return ret
        else:
            PRINT_DISABLED_FLAGDEC = not get_argflag(
                '--noinform', help_='does not print disabled flag decorators')
            if not quiet and PRINT_DISABLED_FLAGDEC:
                #print('\n~~~ %s ~~~' % flag)
                print('~~~ %s ~~~' % flags[0])
    meta_util_six.set_funcname(GaurdWrapper, meta_util_six.get_funcname(func))
    return GaurdWrapper


@profile
def argparse_dict(default_dict_, lbl=None, verbose=None,
                  only_specified=False, force_keys={}, type_hint=None,
                  alias_dict={}):
    r"""
    Gets values for a dict based on the command line

    Args:
        default_dict_ (?):
        only_specified (bool): if True only returns keys that are specified on commandline. no defaults.

    Returns:
        dict_: dict_ -  a dictionary

    CommandLine:
        python -m utool.util_arg --test-argparse_dict
        python -m utool.util_arg --test-argparse_dict --foo=3
        python -m utool.util_arg --test-argparse_dict --flag1
        python -m utool.util_arg --test-argparse_dict --flag2
        python -m utool.util_arg --test-argparse_dict --noflag2
        python -m utool.util_arg --test-argparse_dict --thresh=43
        python -m utool.util_arg --test-argparse_dict --bins=-10
        python -m utool.util_arg --test-argparse_dict --bins=-10 --only-specified --helpx

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> default_dict_ = {
        ...    'bins': 8,
        ...    'foo': None,
        ...    'flag1': False,
        ...    'flag2': True,
        ...    'max': 0.2,
        ...    'neg': -5,
        ...    'thresh': -5.333,
        ... }
        >>> # execute function
        >>> only_specified = ut.get_argflag('--only-specified')
        >>> dict_ = argparse_dict(default_dict_, only_specified=only_specified)
        >>> # verify results
        >>> result = ut.repr4(dict_, sorted_=True)
        >>> print(result)
    """
    if verbose is None:
        verbose = VERBOSE_ARGPARSE
    def make_argstrs(key, prefix_list):
        for prefix in prefix_list:
            yield prefix + key
            yield prefix + key.replace('-', '_')
            yield prefix + key.replace('_', '-')

    def get_dictkey_cmdline_val(key, default, type_hint):
        # see if the user gave a commandline value for this dict key
        defaulttype_ = None if default is None else type(default)
        if type_hint is None:
            type_ = defaulttype_
        elif isinstance(type_hint, dict):
            type_ = type_hint.get(key, defaulttype_)
        elif isinstance(type_hint, type):
            type_ = type_hint
        else:
            raise NotImplementedError('Unknown type of type_hint=%r' % (type_hint,))
        was_specified = False
        if isinstance(default, bool):
            val = default
            if default is True:
                falsekeys = list(set(make_argstrs(key, ['--no', '--no-'])))
                notval, was_specified = get_argflag(falsekeys, return_specified=True)
                val = not notval
                if not was_specified:
                    truekeys = list(set(make_argstrs(key, ['--'])))
                    val_, was_specified = get_argflag(truekeys, return_specified=True)
                    if was_specified:
                        val = val_
            elif default is False:
                truekeys = list(set(make_argstrs(key, ['--'])))
                val, was_specified = get_argflag(truekeys, return_specified=True)
        else:
            argtup = list(set(make_argstrs(key, ['--'])))
            #if key == 'species':
            #    import utool as ut
            #    ut.embed()
            val, was_specified = get_argval(argtup, type_=type_,
                                            default=default,
                                            return_specified=True)
        return val, was_specified

    dict_  = {}
    num_specified = 0
    for key, default in six.iteritems(default_dict_):
        val, was_specified = get_dictkey_cmdline_val(key, default, type_hint)
        if not was_specified:
            alias_keys = meta_util_iter.ensure_iterable(alias_dict.get(key, []))
            for alias_key in alias_keys:
                val, was_specified = get_dictkey_cmdline_val(alias_key, default,
                                                             type_hint)
                if was_specified:
                    break
        if VERBOSE_ARGPARSE:
            if was_specified:
                num_specified += 1
                print('[argparse_dict] Specified key=%r, val=%r' % (key, val))
        #if key == 'foo':
        #    import utool as ut
        #    ut.embed()
        if not only_specified or was_specified or key in force_keys:
            dict_[key] = val
    if VERBOSE_ARGPARSE:
        print('[argparse_dict] num_specified = %r' % (num_specified,))
        print('[argparse_dict] force_keys = %r' % (force_keys,))
    #dict_ = {key: get_dictkey_cmdline_val(key, default) for key, default in
    #six.iteritems(default_dict_)}

    if verbose:
        for key in dict_:
            if dict_[key] != default_dict_[key]:
                print('[argparse_dict] GOT ARGUMENT: cfgdict[%r] = %r' % (key, dict_[key]))

    do_helpx = get_argflag('--helpx',
                           help_='Specifies that argparse_dict should print help and quit')

    if get_argflag(('--help', '--help2')) or do_helpx:
        import utool as ut
        print('COMMAND LINE IS ACCEPTING THESE PARAMS WITH DEFAULTS:')
        if lbl is not None:
            print(lbl)
        #print(ut.align(ut.repr4(dict_, sorted_=True), ':'))
        print(ut.align(ut.repr4(default_dict_, sorted_=True), ':'))
        if do_helpx:
            sys.exit(1)
    return dict_


def get_argv_tail(scriptname, prefer_main=None, argv=None):
    r"""
    gets the rest of the arguments after a script has been invoked hack.
    accounts for python -m scripts.

    Args:
        scriptname (str):

    CommandLine:
        python -m utool.util_arg --test-get_argv_tail

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> from os.path import relpath, dirname
        >>> scriptname = 'utool.util_arg'
        >>> prefer_main = False
        >>> argv=['python', '-m', 'utool.util_arg', '--test-get_argv_tail']
        >>> tail = get_argv_tail(scriptname, prefer_main, argv)
        >>> # hack
        >>> tail[0] = ut.ensure_unixslash(relpath(tail[0], dirname(dirname(ut.__file__))))
        >>> result = ut.repr2(tail)
        >>> print(result)
        ['utool/util_arg.py', '--test-get_argv_tail']

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_arg import *  # NOQA
        >>> import utool as ut
        >>> from os.path import relpath, dirname
        >>> scriptname = 'utprof.py'
        >>> prefer_main = True
        >>> argv=['utprof.py', '-m', 'utool', '--tf', 'get_argv_tail']
        >>> tail = get_argv_tail(scriptname, prefer_main, argv)
        >>> # hack
        >>> tail[0] = ut.ensure_unixslash(relpath(tail[0], dirname(dirname(ut.__file__))))
        >>> result = ut.repr2(tail)
        >>> print(result)
        ['utool/__main__.py', '--tf', 'get_argv_tail']
    """
    if argv is None:
        argv = sys.argv
    import utool as ut
    modname = ut.get_argval('-m', help_='specify module name to profile', argv=argv)
    if modname is not None:
        # hack to account for -m scripts
        modpath = ut.get_modpath(modname, prefer_main=prefer_main)
        argvx = argv.index(modname) + 1
        argv_tail = [modpath] + argv[argvx:]
    else:
        try:
            argvx = argv.index(scriptname)
        except ValueError:
            for argvx, arg in enumerate(argv):
                # HACK
                if scriptname in arg:
                    break
        argv_tail = argv[(argvx + 1):]
    return argv_tail


def get_cmdline_varargs(argv=None):
    """
    Returns positional args specified directly after the scriptname
    and before any args starting with '-' on the commandline.
    """
    if argv is None:
        argv = sys.argv
    scriptname = argv[0]
    if scriptname == '':
        # python invoked by iteself
        pos_start = 0
        pos_end = 0
    else:
        pos_start = pos_end = 1
        for idx in range(pos_start, len(argv)):
            if argv[idx].startswith('-'):
                pos_end = idx
                break
        else:
            pos_end = len(argv)
    cmdline_varargs = argv[pos_start:pos_end]
    return cmdline_varargs


get_varargs = get_cmdline_varargs


# alias
parse_dict_from_argv = argparse_dict
get_dict_vals_from_commandline = argparse_dict

# Do I like these more?
# varargs = get_varargs
argval = get_argval
argflag = get_argflag


def argval(key, default=None, type=None, smartcast=True, return_exists=False,
           argv=None):
    """
    alias for get_argval

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> import sys
        >>> argv = ['--aids=[1,2,3]']
        >>> value = ut.argval('--aids', default=[1, 2], argv=argv)
        >>> assert isinstance(value, list)
        >>> value2 = ut.argval('--aids', smartcast=False, argv=argv)
        >>> assert isinstance(value2, str)
        >>> value2 = ut.argval('--aids', smartcast=True, argv=argv)
        >>> assert isinstance(value2, list)
    """
    defaultable_types = (tuple, list, int, float)
    if type is None and isinstance(default, defaultable_types):
        type = builtins.type(default)
    return get_argval(key, type_=type, default=default,
                      return_was_specified=return_exists, smartcast=smartcast,
                      argv=argv)


VERBOSE_ARGPARSE = get_argflag(
    ('--verbose-argparse', '--verb-argparse', '--verb-arg', '--verbarg'),
    help_='debug util_arg')


#if __name__ == '__main__':
#    """
#    CommandLine:
#        python utool/util_arg.py --test-autogen_argparse2
#    """
#    from utool import util_tests
#    util_tests.doctest_funcs([autogen_argparse2])
if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_arg
        python -m utool.util_arg --allexamples
        python -m utool.util_arg --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
