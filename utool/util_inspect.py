# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import inspect
import os
import os.path
import re
import six
import sys
import types
from os.path import dirname
from six.moves import builtins
from collections import OrderedDict
from six.moves import range, zip  # NOQA
from utool import util_regex
from utool import util_arg
from utool import util_inject
from utool._internal import meta_util_six
print, rrr, profile = util_inject.inject2(__name__, '[inspect]')


VERBOSE_INSPECT, VERYVERB_INSPECT = util_arg.get_module_verbosity_flags('inspect')


LIB_PATH = dirname(os.__file__)


@profile
def check_module_usage(modpath_partterns):
    """
    Args:
        modpath_partterns (?):

    CommandLine:
        python -m utool.util_inspect --exec-check_module_usage --show
        utprof.py -m utool.util_inspect --exec-check_module_usage --show
        python -m utool.util_inspect --exec-check_module_usage --pat="['auto*', 'user_dialogs.py', 'special_query.py', 'qt_inc_automatch.py', 'devcases.py']"

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> modpath_partterns = ['_grave*']
        >>> modpath_partterns = ['auto*', 'user_dialogs.py', 'special_query.py', 'qt_inc_automatch.py', 'devcases.py']
        >>> result = check_module_usage(modpath_partterns)
        >>> print(result)
    """
    import utool as ut
    dpath = '~/code/ibeis/ibeis/algo/hots'
    modpaths = ut.flatten([ut.glob(dpath, pat) for pat in modpath_partterns])
    modnames = ut.lmap(ut.get_modname_from_modpath, modpaths)

    importance_dict = {}

    # HACK: ut.parfor
    # returns a 0 lenth iterator so the for loop is never run uses code
    # introspection to determine the content of the for loop body executes code
    # using the values of the local variables in a parallel / distributed
    # context.
    cache = {}

    for modname, modpath in zip(modnames, modpaths):
        pattern = '\\b' + modname + '\\b',
        found_fpath_list, found_lines_list = ut.grep_projects(pattern, new=True, verbose=False, cache=cache)
        parent_modnames = ut.lmap(ut.get_modname_from_modpath, found_fpath_list)
        parent_numlines = ut.lmap(len, found_lines_list)
        importance = dict(zip(parent_modnames, parent_numlines))
        ut.delete_keys(importance, modnames)
        importance_dict[modname] = importance

    print('importance_dict = %s' % (ut.repr3(importance_dict),))
    combo = reduce(ut.dict_union, importance_dict.values())
    print('combined %s' % (ut.repr3(combo),))
    # print(ut.repr3(found_fpath_list))


def help_members(obj):
    r"""
    Args:
        obj (class or module):

    CommandLine:
        python -m utool.util_inspect --exec-help_members

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> obj = ut.DynStruct
        >>> result = help_members(obj)
        >>> print(result)
    """
    import utool as ut
    attr_list = [getattr(obj, attrname) for attrname in dir(obj)]
    type2_items = ut.group_items(attr_list, list(map(ut.type_str, map(type, attr_list))))
    memtypes = ['instancemethod']  # , 'method-wrapper']
    func_mems = ut.dict_subset(type2_items, memtypes, [])
    #other_mems = ut.delete_keys(type2_items.copy(), memtypes)

    func_list = ut.flatten(func_mems.values())
    defsig_list = []
    num_unbound_args_list = []
    num_args_list = []
    for func in func_list:
        #args = ut.get_func_argspec(func).args
        argspec = ut.get_func_argspec(func)
        args = argspec.args
        unbound_args = get_unbound_args(argspec)
        defsig = ut.func_defsig(func)
        defsig_list.append(defsig)
        num_unbound_args_list.append(len(unbound_args))
        num_args_list.append(len(args))

    group = ut.hierarchical_group_items(defsig_list, [num_unbound_args_list, num_args_list])
    print(repr(obj))
    print(ut.repr3(group, strvals=True))


def get_dev_hints():
    VAL_FIELD = util_regex.named_field('val', '.*')
    VAL_BREF = util_regex.bref_field('val')

    registered_hints = OrderedDict([
        # General IBEIS hints
        ('ibs.*'   , ('ibeis.IBEISController', 'image analysis api')),
        ('testres', ('ibeis.TestResult', 'test result object')),
        ('qreq_'   , ('ibeis.QueryRequest', 'query request object with hyper-parameters')),
        ('cm'  , ('ibeis.ChipMatch', 'object of feature correspondences and scores')),
        ('qparams*', ('ibeis.QueryParams', 'query hyper-parameters')),
        ('qaid2_cm.*'   , ('dict', 'dict of ``ChipMatch`` objects')),
        ('vecs'    , ('ndarray[uint8_t, ndim=2]', 'descriptor vectors')),
        ('maws'    , ('ndarray[float32_t, ndim=1]', 'multiple assignment weights')),
        ('words'   , ('ndarray[uint8_t, ndim=2]', 'aggregate descriptor cluster centers')),
        ('word'    , ('ndarray[uint8_t, ndim=1]', 'aggregate descriptor cluster center')),
        ('rvecs'   , ('ndarray[uint8_t, ndim=2]', 'residual vector')),
        ('fm', ('list', 'list of feature matches as tuples (qfx, dfx)')),
        ('fs', ('list', 'list of feature scores')),
        ('aid_list' , ('list', 'list of annotation rowids')),
        ('aids' ,     ('list', 'list of annotation rowids')),
        ('aid_list[0-9]' , ('list', 'list of annotation ids')),
        ('ensure' , ('bool', 'eager evaluation if True')),
        ('qaid'    , ('int', 'query annotation id')),
        ('aid[0-9]?', ('int', 'annotation id')),
        ('daids'   , ('list', 'database annotation ids')),
        ('qaids'   , ('list', 'query annotation ids')),
        ('use_cache', ('bool', 'turns on disk based caching')),
        ('qreq_vsmany_', ('QueryRequest', 'persistant vsmany query request')),
        ('qnid'    , ('int', 'query name id')),
        #
        ('gfpath[0-9]?' , ('str', 'image file path string')),
        ('path[0-9]?' , ('str', 'path to file or directory')),
        ('n'    , ('int', '')),
        ('ext'    , ('str', 'extension')),
        ('_path' , ('str', 'path string')),
        ('path_' , ('str', 'path string')),
        ('.*_dpath' , ('str', 'directory path string')),
        ('.*_fpath' , ('str', 'file path string')),
        ('bbox' , ('tuple', 'bounding box in the format (x, y, w, h)')),
        ('theta' , ('float', 'angle in radians')),
        ('ori_thresh' , ('float', 'angle in radians')),
        ('xy_thresh_sqrd' , ('float', '')),
        ('xy_thresh' , ('float', '')),
        ('scale_thresh' , ('float', '')),

        # Pipeline hints
        ('qaid2_nns',
         ('dict', 'maps query annotid to (qfx2_idx, qfx2_dist)')),

        ('qaid2_nnvalid0',
         ('dict',
          'maps query annotid to qfx2_valid0')),

        ('qfx2_valid0',
         ('ndarray',
          'maps query feature index to K matches non-impossibility flags')),

        ('filt2_weights',
         ('dict', 'maps filter names to qfx2_weight ndarray')),

        ('qaid2_filtweights',
         ('dict',
          'mapping to weights computed by filters like lnnbnn and ratio')),

        ('qaid2_nnfiltagg',
         ('dict',
          'maps to nnfiltagg - tuple(qfx2_score, qfx2_valid)')),

        ('qaid2_nnfilts',
         ('dict', 'nonaggregate feature scores and validities for each feature NEW')),

        ('nnfiltagg', ('tuple', '(qfx2_score_agg, qfx2_valid_agg)')),
        ('nnfilts', ('tuple', '(filt_list, qfx2_score_list, qfx2_valid_list)')),

        ('qfx2_idx', ('ndarray[int32_t, ndims=2]', 'mapping from query feature index to db neighbor index')),

        ('K'       , ('int', None)),
        ('Knorm'   , ('int', None)),

        # SMK Hints
        ('smk_alpha',  ('float', 'selectivity power')),
        ('smk_thresh', ('float', 'selectivity threshold')),
        ('query_sccw', ('float', 'query self-consistency-criterion')),
        ('data_sccw', ('float', 'data self-consistency-criterion')),
        ('invindex', ('InvertedIndex', 'object for fast vocab lookup')),

        # Plotting hints
        ('[qd]?rchip[0-9]?', ('ndarray[uint8_t, ndim=2]', 'rotated annotation image data')),
        ('[qd]?chip[0-9]?', ('ndarray[uint8_t, ndim=2]', 'annotation image data')),
        ('kp', ('ndarray[float32_t, ndim=1]', 'a single keypoint')),
        ('[qd]?kpts[0-9]?', ('ndarray[float32_t, ndim=2]', 'keypoints')),
        ('[qd]?vecs[0-9]?', ('ndarray[uint8_t, ndim=2]', 'descriptor vectors')),
        ('H', ('ndarray[float64_t, ndim=2]', 'homography/perspective matrix')),
        ('invV_mats2x2', ('ndarray[float32_t, ndim=3]',  'keypoint shapes')),
        ('invVR_mats2x2', ('ndarray[float32_t, ndim=3]', 'keypoint shape and rotations')),
        ('invV_mats', ('ndarray[float32_t, ndim=3]',  'keypoint shapes (possibly translation)')),
        ('invVR_mats', ('ndarray[float32_t, ndim=3]', 'keypoint shape and rotations (possibly translation)')),
        ('img\d*', ('ndarray[uint8_t, ndim=2]', 'image data')),
        ('img_in', ('ndarray[uint8_t, ndim=2]', 'image data')),
        ('arr', ('ndarray', '')),
        ('arr_', ('ndarray', '')),
        ('X', ('ndarray', 'data')),
        ('y', ('ndarray', 'labels')),
        ('imgBGR', ('ndarray[uint8_t, ndim=2]', 'image data in opencv format (blue, green, red)')),
        ('pnum', ('tuple', 'plot number')),
        ('fnum', ('int', 'figure number')),
        ('title', ('str', '')),
        ('text', ('str', '')),
        ('text_', ('str', '')),

        # Matching Hints
        ('ratio_thresh'       , ('float', None)),

        # utool hints
        ('func'           , ('function', 'live python function')),
        ('funcname'       , ('str', 'function name')),
        ('modname'        , ('str', 'module name')),
        ('argname_list'   , ('str', 'list of argument names')),
        ('return_name'    , ('str', 'return variable name')),
        ('dict_'          , ('dict_', 'a dictionary')),
        ('examplecode'    , ('str', None)),

        # Numpy Hints
        ('shape'    , ('tuple', 'array dimensions')),
        ('chipshape'    , ('tuple', 'height, width')),
        ('rng'    , ('RandomState', 'random number generator')),

        # Opencv hings
        ('dsize'    , ('tuple', 'width, height')),
        ('chipsize'    , ('tuple', 'width, height')),

        # Standard Python Hints for my coding style
        ('.*_fn' , ('func', None)),
        ('str_' , ('str', None)),
        ('num_.*' , ('int', None)),
        ('.*_str' , ('str', None)),
        ('.*_?list_?' , ('list', None)),
        ('.*_?dict_?' , ('dict', None)),
        ('dict_?\d?' , ('dict', None)),
        ('.*_tup' , ('tuple', None)),
        ('.*_sublist' , ('list', None)),
        ('fpath[0-9]?' , ('str', 'file path string')),
        ('chip[A-Z]*' , ('ndarray', 'cropped image')),
        ('verbose', ('bool', 'verbosity flag')),

        # Other hints for my coding style
        ('wx2_'    , ('dict', None)),
        ('qfx2_' + VAL_FIELD,
         ('ndarray',
          'mapping from query feature index to ' + VAL_BREF)),
        ('.*x2_.*' , ('ndarray', None)),
        ('.+[^3]2_.*'  , ('dict', None)),
        ('dpath'  , ('str', 'directory path')),
        ('dname'  , ('str', 'directory name')),
        ('fpath'  , ('str', 'file path')),
        ('fname'  , ('str', 'file name')),
        ('pattern'  , ('str', '')),
    ])
    return registered_hints


def infer_arg_types_and_descriptions(argname_list, defaults):
    """

    Args:
        argname_list (list):
        defaults (list):

    Returns:
        tuple : (argtype_list, argdesc_list)

    CommandLine:
        python -m utool.util_inspect --test-infer_arg_types_and_descriptions

    Ignore:
        python -c "import utool; print(utool.auto_docstr('ibeis.algo.hots.pipeline', 'build_chipmatches'))"

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> argname_list = ['ibs', 'qaid', 'fdKfds', 'qfx2_foo']
        >>> defaults = None
        >>> tup = utool.infer_arg_types_and_descriptions(argname_list, defaults)
        >>> argtype_list, argdesc_list, argdefault_list, hasdefault_list = tup
    """
    #import utool as ut
    from utool import util_dev

    # hacks for IBEIS
    if True or util_dev.is_developer():
        registered_hints = get_dev_hints()
        # key = regex pattern
        # val = hint=tuple(type_, desc_)

    if defaults is None:
        defaults = []
    default_types = [type(val).__name__.replace('NoneType', 'None') for val in defaults]
    num_defaults = len(defaults)
    num_nodefaults = len(argname_list) - num_defaults
    argtype_list = ['?'] * (num_nodefaults) + default_types
    # defaults aligned with argtype_list and argdesc_list
    argdefault_list = [None] * num_nodefaults + list(defaults)
    hasdefault_list = [False] * num_nodefaults + [True] * num_defaults

    argdesc_list = ['' for _ in range(len(argname_list))]

    # use hints to build better docstrs
    for argx in range(len(argname_list)):
        #if argtype_list[argx] == '?' or argtype_list[argx] == 'None':
        argname = argname_list[argx]
        if argname is None:
            #print('warning argname is None')
            continue
        for regex, hint in six.iteritems(registered_hints):
            matchobj = re.match('^' + regex + '$', argname, flags=re.MULTILINE | re.DOTALL)
            if matchobj is not None:
                type_ = hint[0]
                desc_ = hint[1]
                if type_ is not None:
                    if argtype_list[argx] == '?' or argtype_list[argx] == 'None':
                        argtype_list[argx] = type_
                if desc_ is not None:
                    desc_ = matchobj.expand(desc_)
                    argdesc_list[argx] = ' ' + desc_
                break
    # append defaults to descriptions
    for argx in range(len(argdesc_list)):
        if hasdefault_list[argx]:
            if isinstance(argdefault_list[argx], types.ModuleType):
                defaultrepr = argdefault_list[argx].__name__
            else:
                defaultrepr = repr(argdefault_list[argx])
            #import utool as ut
            #ut.embed()
            argdesc_list[argx] += '(default = %s)' % (defaultrepr,)
    return argtype_list, argdesc_list, argdefault_list, hasdefault_list


def get_module_owned_functions(module):
    """
    Replace with iter_module_doctesable (but change that name to be something better)

    returns functions actually owned by the module
    module = vtool.distance
    """
    import utool as ut
    list_ = []
    for key, val in ut.iter_module_doctestable(module):
        if hasattr(val, '__module__'):
            belongs = val.__module__ == module.__name__
        elif hasattr(val, 'func_globals'):
            belongs = val.func_globals['__name__'] == module.__name__
        if belongs:
            list_.append(val)
    return list_


def iter_module_doctestable(module, include_funcs=True, include_classes=True,
                            include_methods=True,
                            include_builtin=True,
                            include_inherited=False,
                            debug_key=None):
    r"""
    Yeilds doctestable live object form a modules

    TODO: change name to iter_module_members
    Replace with iter_module_doctesable (but change that name to be something
    better)

    Args:
        module (module): live python module
        include_funcs (bool):
        include_classes (bool):
        include_methods (bool):
        include_builtin (bool): (default = True)
        include_inherited (bool): (default = False)

    Yeilds:
        tuple (str, callable): (funcname, func) doctestable

    CommandLine:
        python -m utool --tf iter_module_doctestable --modname=ibeis.algo.hots.chip_match
        python -m utool --tf iter_module_doctestable --modname=ibeis.control.IBEISControl
        python -m utool --tf iter_module_doctestable --modname=ibeis.control.SQLDatabaseControl
        python -m utool --tf iter_module_doctestable --modname=ibeis.control.manual_annot_funcs
        python -m utool --tf iter_module_doctestable --modname=ibeis.control.manual_annot_funcs
        python -m utool --tf iter_module_doctestable --modname=ibeis.expt.test_result
        python -m utool --tf iter_module_doctestable --modname=utool.util_progress --debug-key=build_msg_fmtstr_time2
        python -m utool --tf iter_module_doctestable --modname=utool.util_progress --debug-key=ProgressIter

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *   # NOQA
        >>> import utool as ut
        >>> modname = ut.get_argval('--modname', type_=str, default=None)
        >>> kwargs = ut.argparse_funckw(iter_module_doctestable)
        >>> module = ut.util_tests if modname is None else ut.import_modname(modname)
        >>> #debug_key = ut.get_argval('--debugkey', type_=str, default=None)
        >>> doctestable_list = list(iter_module_doctestable(module, **kwargs))
        >>> func_names = sorted(ut.get_list_column(doctestable_list, 0))
        >>> print(ut.list_str(func_names))
    """
    import ctypes

    types.BuiltinFunctionType
    valid_func_types = [
        types.FunctionType, types.BuiltinFunctionType, classmethod,
        staticmethod,
        #types.MethodType, types.BuiltinMethodType,
    ]
    if include_builtin:
        valid_func_types += [
            types.BuiltinFunctionType
        ]

    if six.PY2:
        valid_class_types = (types.ClassType,  types.TypeType,)
    else:
        valid_class_types = six.class_types

    scalar_types = ([dict, list, tuple, set, frozenset, bool, float, int] +
                    list(six.string_types))
    scalar_types += list(six.string_types)
    other_types = [functools.partial, types.ModuleType,
                   ctypes.CDLL]
    if six.PY2:
        other_types += [types.InstanceType]

    invalid_types = tuple(scalar_types + other_types)
    valid_func_types = tuple(valid_func_types)

    #modpath = ut.get_modname_from_modpath(module.__file__)

    for key, val in six.iteritems(module.__dict__):
        # <DEBUG>
        if debug_key is not None and key == debug_key:
            import utool as ut
            ut.embed()
        # </DEBUG>
        if hasattr(val, '__module__'):
            # HACK: todo. figure out true parent module
            if val.__module__ == 'numpy':
                continue
        if val is None:
            pass
        elif isinstance(val, valid_func_types):
            if include_funcs:
                if not include_inherited and not is_defined_by_module(val, module):
                    continue
                yield key, val
        elif isinstance(val, valid_class_types):
            class_ = val
            if not include_inherited and not is_defined_by_module(class_, module):
                continue
            if include_classes:
                # Yield the class itself
                yield key, val
            if include_methods:
                # Yield methods of the class
                for subkey, subval in six.iteritems(class_.__dict__):
                    if isinstance(subval, property):
                        subval = subval.fget
                    # <DEBUG>
                    if debug_key is not None and subkey == debug_key:
                        import utool as ut
                        ut.embed()
                    # </DEBUG>
                    # Unbound methods are still typed as functions
                    if isinstance(subval, valid_func_types):
                        if not include_inherited and not is_defined_by_module(subval, module):
                            continue
                        if isinstance(subval, (staticmethod)):
                            subval.__func__.__ut_parent_class__ = class_
                        if not isinstance(subval, types.BuiltinFunctionType) and not isinstance(subval, (classmethod, staticmethod)):
                            # HACK: __ut_parent_class__ lets util_test have
                            # more info on the func should return extra info
                            # instead
                            subval.__ut_parent_class__ = class_
                        yield subkey, subval
                    elif isinstance(val, invalid_types):
                        pass
                    else:
                        if util_arg.VERBOSE:
                            print('[util_inspect] WARNING module %r class %r:' % (module, class_,))
                            print(' * Unknown if testable val=%r' % (val))
                            print(' * Unknown if testable type(val)=%r' % type(val))
        elif isinstance(val, invalid_types):
            pass
        else:
            #import utool as ut
            if util_arg.VERBOSE:
                print('[util_inspect] WARNING in module %r:' % (module,))
                print(' * Unknown if testable val=%r' % (val))
                print(' * Unknown if testable type(val)=%r' % type(val))


def is_defined_by_module(item, module):
    """
    Check if item is directly defined by a module.
    This check may be prone to errors.
    """
    flag = False
    if isinstance(item, types.ModuleType):
        if not hasattr(item, '__file__'):
            return False
        item_modpath = os.path.realpath(dirname(item.__file__))
        mod_fpath = module.__file__.replace('.pyc', '.py')
        if not mod_fpath.endswith('__init__.py'):
            return False
        modpath = os.path.realpath(dirname(mod_fpath))
        modpath = modpath.replace('.pyc', '.py')
        return item_modpath.startswith(modpath)
    elif hasattr(item, '_utinfo'):
        # Capture case where there is a utool wrapper
        orig_func = item._utinfo['orig_func']
        flag = is_defined_by_module(orig_func, module)
    else:
        if isinstance(item, staticmethod):
            # static methods are a wrapper around a function
            item = item.__func__
        try:
            func_globals = meta_util_six.get_funcglobals(item)
            if func_globals['__name__'] == module.__name__:
                flag = True
        except  AttributeError:
            if hasattr(item, '__module__'):
                flag = item.__module__ == module.__name__
    return flag


def get_func_modname(func):
    if hasattr(func, '_utinfo'):
        # Capture case where there is a utool wrapper
        orig_func = func._utinfo['orig_func']
        return get_func_modname(orig_func)
    #try:
    func_globals = meta_util_six.get_funcglobals(func)
    modname = func_globals['__name__']
    return modname
    #except  AttributeError:
    #    pass
    #pass


def is_bateries_included(item):
    """
    Returns if a value is a python builtin function

    Args:
        item (object):

    Returns:
        bool: flag

    References:
        http://stackoverflow.com/questions/23149218/check-if-a-python-function-is-builtin

    CommandLine:
        python -m utool._internal.meta_util_six --exec-is_builtin

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool._internal.meta_util_six import *  # NOQA
        >>> item = zip
        >>> flag = is_bateries_included(item)
        >>> result = ('flag = %s' % (str(flag),))
        >>> print(result)
    """
    flag = False
    if hasattr(item, '__call__') and hasattr(item, '__module__'):
        if item.__module__ is not None:
            module = sys.modules[item.__module__]
            if module == builtins:
                flag = True
            elif hasattr(module, '__file__'):
                flag = LIB_PATH == dirname(module.__file__)
    return flag


def list_class_funcnames(fname, blank_pats=['    #']):
    """
    list_class_funcnames

    Args:
        fname (str): filepath
        blank_pats (list): defaults to '    #'

    Returns:
        list: funcname_list

    Example:
        >>> from utool.util_inspect import *  # NOQA
        >>> fname = 'util_class.py'
        >>> blank_pats = ['    #']
        >>> funcname_list = list_class_funcnames(fname, blank_pats)
        >>> print(funcname_list)
    """
    with open(fname, 'r') as file_:
        lines = file_.readlines()
    funcname_list = []

    #full_line_ = ''
    for lx, line in enumerate(lines):
        #full_line_ += line
        if any([line.startswith(pat) for pat in blank_pats]):
            funcname_list.append('')
        if line.startswith('    def '):
            def_x    = line.find('def')
            rparen_x = line.find('(')
            funcname = line[(def_x + 3):rparen_x]
            #print(funcname)
            funcname_list.append(funcname)
    return funcname_list


def list_global_funcnames(fname, blank_pats=['    #']):
    """
    list_global_funcnames

    Args:
        fname (str): filepath
        blank_pats (list): defaults to '    #'

    Returns:
        list: funcname_list

    Example:
        >>> from utool.util_inspect import *  # NOQA
        >>> fname = 'util_class.py'
        >>> blank_pats = ['    #']
        >>> funcname_list = list_global_funcnames(fname, blank_pats)
        >>> print(funcname_list)
    """
    with open(fname, 'r') as file_:
        lines = file_.readlines()
    funcname_list = []

    #full_line_ = ''
    for lx, line in enumerate(lines):
        #full_line_ += line
        if any([line.startswith(pat) for pat in blank_pats]):
            funcname_list.append('')
        if line.startswith('def '):
            def_x    = line.find('def')
            rparen_x = line.find('(')
            funcname = line[(def_x + 3):rparen_x]
            #print(funcname)
            funcname_list.append(funcname)
    return funcname_list


def inherit_kwargs(inherit_func):
    """
    TODO move to util_decor
    inherit_func = inspect_pdfs
    func = encoder.visualize.im_func
    """
    import utool as ut
    keys, is_arbitrary = ut.get_kwargs(inherit_func)
    if is_arbitrary:
        keys += ['**kwargs']
    kwargs_append = '\n'.join(keys)
    #from six.moves import builtins
    #builtins.print(kwargs_block)
    def _wrp(func):
        if func.__doc__ is None:
            func.__doc__ = ''
        # TODO append to kwargs block if it exists
        kwargs_block = 'Kwargs:\n' + ut.indent(kwargs_append)
        func.__doc__ += kwargs_block
        return func
    return _wrp


def filter_valid_kwargs(func, dict_):
    import utool as ut
    keys, is_arbitrary = ut.get_kwargs(func)
    if is_arbitrary:
        valid_dict_ = dict_
    else:
        key_subset = ut.dict_keysubset(dict_, keys)
        valid_dict_ = ut.dict_subset(dict_, key_subset)
    return valid_dict_


def dummy_func(arg1, arg2, arg3=None, arg4=[1, 2, 3], arg5={}, **kwargs):
    """
    test func for kwargs parseing
    """
    foo = kwargs.get('foo', None)
    bar = kwargs.pop('bar', 4)
    foobar = str(foo) + str(bar)
    return foobar


def get_kwdefaults2(func, parse_source=False):
    return get_kwdefaults(func, parse_source=True)


def get_kwdefaults(func, parse_source=False):
    r"""
    Args:
        func (func):

    Returns:
        dict:

    CommandLine:
        python -m utool.util_inspect --exec-get_kwdefaults

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> func = dummy_func
        >>> parse_source = True
        >>> kwdefaults = get_kwdefaults(func, parse_source)
        >>> print('kwdefaults = %s' % (ut.dict_str(kwdefaults),))
    """
    #import utool as ut
    #with ut.embed_on_exception_context:
    argspec = inspect.getargspec(func)
    kwdefaults = {}
    if argspec.args is None or argspec.defaults is None:
        pass
    else:
        args = argspec.args
        defaults = argspec.defaults
        #kwdefaults = OrderedDict(zip(argspec.args[::-1], argspec.defaults[::-1]))
        kwpos = len(args) - len(defaults)
        kwdefaults = OrderedDict(zip(args[kwpos:], defaults))
    if parse_source and argspec.keywords:
        # TODO parse for kwargs.get/pop
        keyword_defaults = parse_func_kwarg_keys(func, with_vals=True)
        for key, val in keyword_defaults:
            assert key not in kwdefaults, 'parsing error'
            kwdefaults[key] = val
    return kwdefaults


def get_argnames(func):
    argspec = inspect.getargspec(func)
    argnames = argspec.args
    return argnames


def get_funcname(func):
    return meta_util_six.get_funcname(func)


def set_funcname(func, newname):
    return meta_util_six.set_funcname(func, newname)


def get_imfunc(func):
    return meta_util_six.get_imfunc(func)


def get_funcglobals(func):
    return meta_util_six.get_funcglobals(func)


def get_funcdoc(func):
    return meta_util_six.get_funcdoc(func)


def get_funcfpath(func):
    return func.func_code.co_filename


def set_funcdoc(func, newdoc):
    return meta_util_six.set_funcdoc(func, newdoc)


def get_docstr(func_or_class):
    """  Get the docstring from a live object """
    import utool as ut
    try:
        docstr_ = func_or_class.func_doc
    except AttributeError:
        docstr_ = func_or_class.__doc__
    if docstr_ is None:
        docstr_ = ''
    docstr = ut.unindent(docstr_)
    return docstr


def prettyprint_parsetree(pt):
    """
    pip install astdump
    pip install codegen
    """
    #import astdump
    import astor
    #import codegen
    #import ast
    #astdump.indented(pt)
    #print(ast.dump(pt, include_attributes=True))
    print(astor.dump(pt))


def find_child_kwarg_funcs(sourcecode, target_kwargs_name='kwargs'):
    r"""
    CommandLine:
        python3 -m utool.util_inspect --exec-find_child_kwarg_funcs

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> sourcecode = ut.codeblock(
            '''
            warped_patch1_list, warped_patch2_list = list(zip(*ut.ichunks(data, 2)))
            interact_patches(labels, warped_patch1_list, warped_patch2_list, flat_metadata, **kwargs)
            import sys
            sys.badcall(**kwargs)
            def foo():
                bar(**kwargs)
                ut.holymoly(**kwargs)
                baz()
                def biz(**kwargs):
                    foo2(**kwargs)
            ''')
        >>> child_funcnamess = ut.find_child_kwarg_funcs(sourcecode)
        >>> print('child_funcnamess = %r' % (child_funcnamess,))
        >>> assert 'foo2' not in child_funcnamess, 'foo2 should not be found'
        >>> assert 'bar' in child_funcnamess, 'bar should be found'

    Notes:
    """
    import ast
    sourcecode = 'from __future__ import print_function\n' + sourcecode
    pt = ast.parse(sourcecode)
    child_funcnamess = []
    debug = False or VERYVERB_INSPECT

    if debug:
        print('\nInput:')
        print('target_kwargs_name = %r' % (target_kwargs_name,))
        print('\nSource:')
        print(sourcecode)
        import astor
        print('\nParse:')
        print(astor.dump(pt))

    class KwargParseVisitor(ast.NodeVisitor):
        """
        TODO: understand ut.update_existing and dict update
        """
        def visit_FunctionDef(self, node):
            if debug:
                print('\nVISIT FunctionDef node = %r' % (node,))
                print('node.args.kwarg = %r' % (node.args.kwarg,))
            if six.PY2:
                kwarg_name = node.args.kwarg
            else:
                if node.args.kwarg is None:
                    kwarg_name = None
                else:
                    kwarg_name = node.args.kwarg.arg
                #import utool as ut
                #ut.embed()
            if kwarg_name != target_kwargs_name:
                # target kwargs is still in scope
                ast.NodeVisitor.generic_visit(self, node)

        def visit_Call(self, node):
            if debug:
                print('\nVISIT Call node = %r' % (node,))
                #print(ut.dict_str(node.__dict__,))
            if isinstance(node.func, ast.Attribute):
                funcname = node.func.value.id + '.' + node.func.attr
            elif isinstance(node.func, ast.Name):
                funcname = node.func.id
            else:
                raise NotImplementedError(
                    'do not know how to parse: node.func = %r' % (node.func,))
            kwargs = node.kwargs
            kwargs_name = None if kwargs is None else kwargs.id
            if kwargs_name == target_kwargs_name:
                child_funcnamess.append(funcname)
            if debug:
                print('funcname = %r' % (funcname,))
                print('kwargs_name = %r' % (kwargs_name,))
            ast.NodeVisitor.generic_visit(self, node)
    try:
        KwargParseVisitor().visit(pt)
    except Exception:
        pass
        #import utool as ut
        #if ut.SUPER_STRICT:
        #    raise
    return child_funcnamess
    #print('child_funcnamess = %r' % (child_funcnamess,))


def is_valid_python(code, reraise=True, ipy_magic_workaround=False):
    """
    References:
        http://stackoverflow.com/questions/23576681/python-check-syntax
    """
    import ast
    try:
        if ipy_magic_workaround:
            code = '\n'.join(['pass' if re.match(r'\s*%[a-z]*', line) else line for line in code.split('\n')])
        ast.parse(code)
    except SyntaxError:
        if reraise:
            import utool as ut
            print('Syntax Error')
            ut.print_python_code(code)
            raise
        return False
    return True


def parse_return_type(sourcecode):
    r"""

    parse_return_type

    Args:
        sourcecode (?):

    Returns:
        tuple: (return_type, return_name, return_header)

    Ignore:
        testcase
        automated_helpers query_vsone_verified

    CommandLine:
        python -m utool.util_inspect --exec-parse_return_type
        python -m utool.util_inspect --test-parse_return_type


    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> sourcecode = ut.codeblock(
        ... 'def foo(tmp=False):\n'
        ... '    bar = True\n'
        ... '    return bar\n'
        ... )
        >>> returninfo = parse_return_type(sourcecode)
        >>> result = ut.repr2(returninfo)
        >>> print(result)
        ('?', 'bar', 'Returns', '')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> sourcecode = ut.codeblock(
        ... 'def foo(tmp=False):\n'
        ... '    return True\n'
        ... )
        >>> returninfo = parse_return_type(sourcecode)
        >>> result = ut.repr2(returninfo)
        >>> print(result)
        ('bool', 'True', 'Returns', '')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> sourcecode = ut.codeblock(
        ... 'def foo(tmp=False):\n'
        ... '    for i in range(2): \n'
        ... '        yield i\n'
        ... )
        >>> returninfo = parse_return_type(sourcecode)
        >>> result = ut.repr2(returninfo)
        >>> print(result)
        ('?', 'i', 'Yields', '')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> sourcecode = ut.codeblock(
        ... 'def foo(tmp=False):\n'
        ... '    if tmp is True:\n'
        ... '        return (True, False)\n'
        ... '    elif tmp is False:\n'
        ... '        return 1\n'
        ... '    else:\n'
        ... '        bar = baz()\n'
        ... '        return bar\n'
        ... )
        >>> returninfo = parse_return_type(sourcecode)
        >>> result = ut.repr2(returninfo)
        >>> print(result)
        ('tuple', '(True, False)', 'Returns', '')
    """
    import ast
    return_type, return_name, return_header = (None, None, None)

    if sourcecode is None:
        return return_type, return_name, return_header, None

    sourcecode = 'from __future__ import print_function\n' + sourcecode

    pt = ast.parse(sourcecode)
    import utool as ut
    debug = ut.get_argflag('--debug-parse-return')
    #debug = True

    if debug:
        import astor
        print('\nSource:')
        print(sourcecode)
        print('\nParse:')
        print(astor.dump(pt))
        print('... starting')

    def print_visit(type_, node):
        if debug:
            import utool as ut
            print('+---')
            print('\nVISIT %s node = %r' % (type_, node,))
            print('node.__dict__ = ' + ut.repr2(node.__dict__, nl=True))
            print('L___')

    def get_node_name_and_type(node):
        if isinstance(node, ast.Tuple):
            tupnode_list = node.elts
            tupleid = '(%s)' % (', '.join([str(get_node_name_and_type(tupnode)[1]) for tupnode in tupnode_list]))
            node_type = 'tuple'
            node_name = tupleid
        elif isinstance(node, ast.Dict):
            node_type = 'dict'
            node_name = None
        elif isinstance(node, ast.Name):
            node_name = node.id
            node_type = '?'
            if node_name in ['True', 'False']:
                node_type = 'bool'
            elif node_name == 'None':
                node_type = 'None'
        elif six.PY3 and isinstance(node, ast.NameConstant):
            node_name = str(node.value)
            node_type = '?'
            if node_name in ['True', 'False', True, False]:
                node_type = 'bool'
            elif node_name in ['None', None]:
                node_type = 'None'
        else:
            node_name = None
            node_type = '?'
            #node_type = 'ADD_TO_GET_NODE_NAME_AND_TYPE: ' + str(type(node.value))
        return node_type, node_name

    class ReturnVisitor(ast.NodeVisitor):
        def init(self):
            self.found_nodes = []
            self.return_header = None

        def visit_FunctionDef(self, node):
            print_visit('FunctionDef', node)
            # TODO: ignore subfunction return types
            ast.NodeVisitor.generic_visit(self, node)

        def visit_Return(self, node):
            print_visit('Return', node)
            ast.NodeVisitor.generic_visit(self, node)
            return_value = node.value
            print_visit('ReturnValue', return_value)
            self.found_nodes.append(return_value)
            self.return_header = 'Returns'

        def visit_Yield(self, node):
            print_visit('Yield', node)
            ast.NodeVisitor.generic_visit(self, node)
            return_value = node.value
            print_visit('YieldValue', return_value)
            self.found_nodes.append(return_value)
            self.return_header = 'Yields'
    try:
        self = ReturnVisitor()
        self.init()
        self.visit(pt)
        return_header = self.return_header
        if len(self.found_nodes) > 0:
            # hack rectify multiple return values
            node = self.found_nodes[0]
            return_type, return_name = get_node_name_and_type(node)
        else:
            return_name = None
            return_type = 'None'

    except Exception:
        if debug:
            raise

    return_desc = ''

    if return_type == '?':
        tup = infer_arg_types_and_descriptions([return_name], [])
        argtype_list, argdesc_list, argdefault_list, hasdefault_list = tup
        return_type = argtype_list[0]
        return_desc = argdesc_list[0]

    return return_type, return_name, return_header, return_desc


#def parse_return_type_OLD(sourcecode):
#    import utool as ut
#    import ast
#    if ut.VERBOSE:
#        print('[utool] parsing return types')

#    if sourcecode is None:
#        return_type, return_name, return_header = (None, None, None)
#        return return_type, return_name, return_header, None

#    #source_lines = sourcecode.splitlines()
#    sourcecode = 'from __future__ import print_function\n' + sourcecode
#    try:
#        pt = ast.parse(sourcecode)
#    except Exception:
#        return_type, return_name, return_header = (None, None, None)
#        #raise
#        return return_type, return_name, return_header, None
#        #print(sourcecode)
#        #ut.printex(ex, 'Error Parsing')

#    assert isinstance(pt, ast.Module), str(type(pt))

#    Try = ast.Try if six.PY3 else ast.TryExcept

#    def find_function_nodes(pt):
#        function_nodes = []
#        for node in pt.body:
#            if isinstance(node, ast.FunctionDef):
#                function_nodes.append(node)
#        return function_nodes

#    function_nodes = find_function_nodes(pt)
#    assert len(function_nodes) == 1
#    func_node = function_nodes[0]

#    def find_return_node(node):
#        if isinstance(node, list):
#            candidates = []
#            node_list = node
#            for node in node_list:
#                candidate = find_return_node(node)
#                if candidate is not None:
#                    candidates.append(candidate)
#            if len(candidates) > 0:
#                return candidates[0]
#        elif isinstance(node, (ast.Return, ast.Yield)):
#            return node
#        elif isinstance(node, (ast.If, Try)):
#            return find_return_node(node.body)
#        else:
#            pass
#            #print(type(node))
#    if ut.VERBOSE:
#        print('[utool] parsing return types')
#    returnnode = find_return_node(func_node.body)
#    # Check return or yeild
#    if isinstance(returnnode, ast.Yield):
#        return_header = 'Yeilds'
#    elif isinstance(returnnode, ast.Return):
#        return_header = 'Returns'
#    else:
#        return_header = None
#    # Get more return info

#    def get_node_name_and_type(node):
#        node_name = None
#        node_type = '?'
#        if node is None:
#            node_type = 'None'
#        elif isinstance(node.value, ast.Tuple):
#            tupnode_list = node.value.elts
#            def get_tuple_membername(tupnode):
#                if hasattr(tupnode, 'id'):
#                    return tupnode.id
#                elif hasattr(tupnode, 'value'):
#                    return 'None'
#                else:
#                    return 'None'
#                pass
#            tupleid = '(%s)' % (', '.join([str(get_tuple_membername(tupnode)) for tupnode in tupnode_list]))
#            node_type = 'tuple'
#            node_name = tupleid
#            #node_name = ast.dump(node)
#        elif isinstance(node.value, ast.Dict):
#            node_type = 'dict'
#            node_name = None
#        elif isinstance(node.value, ast.Name):
#            node_name = node.value.id
#            if node_name == 'True':
#                node_name = 'True'
#                node_type = 'bool'
#        else:
#            #node_type = 'ADD_TO_GET_NODE_NAME_AND_TYPE: ' + str(type(node.value))
#            node_type = '?'
#        return node_type, node_name

#    return_type, return_name = get_node_name_and_type(returnnode)

#    if return_type == '?':
#        tup = infer_arg_types_and_descriptions([return_name], [])
#        argtype_list, argdesc_list, argdefault_list, hasdefault_list = tup
#        return_type = argtype_list[0]
#        return_desc = argdesc_list[0]
#    else:
#        return_desc = ''

#    return return_type, return_name, return_header, return_desc


def exec_func_src(func, globals_=None, locals_=None, key_list=None, sentinal=None):
    """ execs a func and returns requested local vars """
    import utool as ut
    sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True)
    if globals_ is None:
        globals_ = ut.get_parent_globals()
    if locals_ is None:
        locals_ = ut.get_parent_locals()
    if sentinal is not None:
        sourcecode = ut.replace_between_tags(sourcecode, '', sentinal)
    globals_ = globals_.copy()
    if locals_ is not None:
        globals_.update(locals_)
    #six.exec_(sourcecode, globals_, locals_)
    six.exec_(sourcecode, globals_)
    # Draw intermediate steps
    if key_list is None:
        #return locals_
        # TODO autodetermine the key_list from the function vars
        return globals_
    else:
        #var_list = ut.dict_take(locals_, key_list)
        var_list = ut.dict_take(globals_, key_list)
        return var_list


def get_func_sourcecode(func, stripdef=False, stripret=False,
                        strip_docstr=False, strip_comments=False,
                        remove_linenums=None):
    """
    wrapper around inspect.getsource but takes into account utool decorators
    strip flags are very hacky as of now

    Args:
        func (function):
        stripdef (bool):
        stripret (bool): (default = False)
        strip_docstr (bool): (default = False)
        strip_comments (bool): (default = False)
        remove_linenums (None): (default = None)

    CommandLine:
        python -m utool.util_inspect --test-get_func_sourcecode

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> # build test data
        >>> func = get_func_sourcecode
        >>> stripdef = True
        >>> stripret = True
        >>> sourcecode = get_func_sourcecode(func, stripdef)
        >>> # verify results
        >>> print(result)
    """
    import utool as ut
    #try:
    inspect.linecache.clearcache()  # HACK: fix inspect bug
    sourcefile = inspect.getsourcefile(func)
    #except IOError:
    #    sourcefile = None
    if hasattr(func, '_utinfo'):
        #if 'src' in func._utinfo:
        #    sourcecode = func._utinfo['src']
        #else:
        func2 = func._utinfo['orig_func']
        sourcecode = get_func_sourcecode(func2)
    elif sourcefile is not None and sourcefile != '<string>':
        try_limit = 2
        for num_tries in range(try_limit):
            try:
                #print(func)
                sourcecode = inspect.getsource(func)
                break
                #print(sourcecode)
            except (IndexError, OSError, SyntaxError) as ex:
                ut.printex(ex, 'Error getting source',
                           keys=['sourcefile', 'func'])
                if False:
                    # VERY HACK: try to reload the module and get a redefined
                    # version of the function
                    import imp
                    modname = get_func_modname(func)
                    funcname = ut.get_funcname(func)
                    module = sys.modules[modname]
                    # TODO: ut.reload_module()
                    module = imp.reload(module)
                    func = module.__dict__[funcname]
                else:
                    # Fix inspect bug in python2.7
                    inspect.linecache.clearcache()
                if num_tries + 1 != try_limit:
                    tries_left = try_limit - num_tries - 1
                    print('Attempting %d more time(s)' % (tries_left))
                else:
                    raise
    else:
        sourcecode = None
    #orig_source = sourcecode
    #print(orig_source)
    if stripdef:
        # hacky
        sourcecode = ut.unindent(sourcecode)
        #sourcecode = ut.unindent(ut.regex_replace('def [^)]*\\):\n', '', sourcecode))
        #nodef_source = ut.regex_replace('def [^:]*\\):\n', '', sourcecode)
        regex_decor = '^@.' + ut.REGEX_NONGREEDY
        regex_defline = '^def [^:]*\\):\n'
        nodef_source = ut.regex_replace('(' + regex_decor + ')?' + regex_defline, '', sourcecode)
        sourcecode = ut.unindent(nodef_source)
        #print(sourcecode)
        pass
    if stripret:
        r""" \s is a whitespace char """
        return_ = ut.named_field('return', 'return .*$')
        prereturn = ut.named_field('prereturn', r'^\s*')
        return_bref = ut.bref_field('return')
        prereturn_bref = ut.bref_field('prereturn')
        regex = prereturn + return_
        repl = prereturn_bref + 'pass  # ' + return_bref
        #import re
        #print(re.search(regex, sourcecode, flags=re.MULTILINE ))
        #print(re.search('return', sourcecode, flags=re.MULTILINE | re.DOTALL ))
        #print(re.search(regex, sourcecode))
        sourcecode_ = re.sub(regex, repl, sourcecode, flags=re.MULTILINE)
        #print(sourcecode_)
        sourcecode = sourcecode_
        pass
    if strip_docstr or strip_comments:
        # pip install pyminifier
        # References: http://code.activestate.com/recipes/576704/
        #from pyminifier import minification, token_utils
        def remove_docstrings_or_comments(source):
            """
            TODO: commit clean version to pyminifier
            """
            import tokenize
            from six.moves import StringIO
            io_obj = StringIO(source)
            out = ''
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (' ' * (start_col - last_col))
                # Remove comments:
                if strip_comments and token_type == tokenize.COMMENT:
                    pass
                elif strip_docstr and token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                        # This is likely a docstring; double-check we're not inside an operator:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line
            return out
        sourcecode = remove_docstrings_or_comments(sourcecode)
        #sourcecode = minification.remove_comments_and_docstrings(sourcecode)
        #tokens = token_utils.listified_tokenizer(sourcecode)
        #minification.remove_comments(tokens)
        #minification.remove_docstrings(tokens)
        #token_utils.untokenize(tokens)

    if remove_linenums is not None:
        source_lines = sourcecode.strip('\n').split('\n')
        ut.delete_items_by_index(source_lines, remove_linenums)
        sourcecode = '\n'.join(source_lines)
    return sourcecode
    #else:
    #return get_func_sourcecode(func._utinfo['src'])


def get_unbound_args(argspec):
    try:
        args = argspec.args
    except Exception:
        func = argspec
        argspec = get_func_argspec(func)
        args = argspec.args
    args = argspec.args
    defaults = argspec.defaults
    if defaults is not None:
        kwpos = len(args) - len(defaults)
        unbound_args = args[:kwpos]
    else:
        unbound_args = args
    return unbound_args


def get_func_argspec(func):
    """
    wrapper around inspect.getargspec but takes into account utool decorators
    """
    if hasattr(func, '_utinfo'):
        argspec = func._utinfo['orig_argspec']
        return argspec
    if isinstance(func, property):
        func = func.fget
    argspec = inspect.getargspec(func)
    return argspec


def get_kwargs(func):
    """
    Args:
        func (function):

    Returns:
        tuple: keys, is_arbitrary
            keys (list): kwargs keys
            is_arbitrary (bool): has generic **kwargs

    CommandLine:
        python -m utool.util_inspect --test-get_kwargs

    Ignore:
        def func1(a, b, c):
            pass
        def func2(a, b, c, *args):
            pass
        def func3(a, b, c, *args, **kwargs):
            pass
        def func4(a, b=1, c=2):
            pass
        def func5(a, b=1, c=2, *args):
            pass
        def func6(a, b=1, c=2, **kwargs):
            pass
        def func7(a, b=1, c=2, *args, **kwargs):
            pass
        for func in [locals()['func' + str(x)] for x in range(1, 8)]:
            print(inspect.getargspec(func))

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> # build test data
        >>> func = '?'
        >>> result = get_kwargs(func)
        >>> # verify results
        >>> print(result)
    """
    #if argspec.keywords is None:
    import utool as ut
    argspec = inspect.getargspec(func)
    if argspec.defaults is not None:
        num_args = len(argspec.args)
        num_keys = len(argspec.defaults)
        keys = ut.take(argspec.args, range(num_args - num_keys, num_args))
    else:
        keys = []
    is_arbitrary = argspec.keywords is not None
    RECURSIVE = False
    if RECURSIVE and argspec.keywords is not None:
        pass
        # TODO: look inside function at the functions that the kwargs object is being
        # passed to
    return keys, is_arbitrary


def lookup_attribute_chain(attrname, namespace):
    """
        >>> import utool as ut
        >>> globals_ = ut.util_inspect.__dict__
        >>> attrname = 'KWReg.print_defaultkw'
    """
    #subdict = meta_util_six.get_funcglobals(root_func)
    subtup = attrname.split('.')
    subdict = namespace
    for attr in subtup[:-1]:
        subdict = subdict[attr].__dict__
    leaf_attr = subdict[subtup[-1]]
    return leaf_attr


def recursive_parse_kwargs(root_func, path_=None):
    """
    recursive kwargs parser
    FIXME: if docstr indentation is off, this fails

    Args:
        root_func (function):  live python function
        path_ (None): (default = None)

    Returns:
        list:

    CommandLine:
        python -m utool.util_inspect --exec-recursive_parse_kwargs:0
        python -m utool.util_inspect --exec-recursive_parse_kwargs:1
        python -m utool.util_inspect --exec-recursive_parse_kwargs:2 --mod plottool --func draw_histogram

        python -m utool.util_inspect --exec-recursive_parse_kwargs:2 --mod vtool --func ScoreNormalizer.visualize

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> root_func = iter_module_doctestable
        >>> path_ = None
        >>> result = ut.repr2(recursive_parse_kwargs(root_func), nl=1)
        >>> print(result)
        [
            ('include_funcs', True),
            ('include_classes', True),
            ('include_methods', True),
            ('include_builtin', True),
            ('include_inherited', False),
            ('debug_key', None),
        ]


    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> from ibeis.algo.hots import chip_match
        >>> import utool as ut
        >>> root_func = chip_match.ChipMatch.show_ranked_matches
        >>> path_ = None
        >>> result = ut.repr2(recursive_parse_kwargs(root_func))
        >>> print(result)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> modname = ut.get_argval('--mod', type_=str, default='plottool')
        >>> funcname = ut.get_argval('--func', type_=str, default='draw_histogram')
        >>> mod = ut.import_modname(modname)
        >>> root_func = lookup_attribute_chain(funcname, mod.__dict__)
        >>> result = ut.repr2(recursive_parse_kwargs(root_func))
        >>> print(result)
    """
    if VERBOSE_INSPECT:
        print('[inspect] recursive parse kwargs root_func = %r ' % (root_func,))

    import utool as ut
    if path_ is None:
        path_ = []
    if root_func in path_:
        return []
    path_.append(root_func)
    spec = ut.get_func_argspec(root_func)
    # ADD MORE
    kwargs_list = list(ut.get_kwdefaults(root_func, parse_source=False).items())
    #kwargs_list = [(kw,) for kw in  ut.get_kwargs(root_func)[0]]
    sourcecode = ut.get_func_sourcecode(root_func, strip_docstr=True,
                                        stripdef=True)
    kwargs_list += ut.parse_kwarg_keys(sourcecode, with_vals=True)

    def hack_lookup_mod_attrs(attr):
        # HACKS TODO: have find_child_kwarg_funcs infer an attribute is a
        # module / function / type. In the module case, we can import it and
        # look it up.  Maybe args, or returns can help infer type.  Maybe just
        # register some known varnames.  Maybe jedi has some better way to do
        # this.
        if attr == 'ut':
            subdict = ut.__dict__
        elif attr == 'pt':
            import plottool as pt
            subdict = pt.__dict__
        else:
            subdict = None
        return subdict

    def resolve_attr_subfunc(subfunc_name):
        # look up attriute chain
        #subdict = root_func.func_globals
        subdict = meta_util_six.get_funcglobals(root_func)
        subtup = subfunc_name.split('.')
        try:
            subdict = lookup_attribute_chain(subfunc_name, subdict)
        except (KeyError, TypeError):
            for attr in subtup[:-1]:
                try:
                    subdict = subdict[attr].__dict__
                except (KeyError, TypeError):
                    # limited support for class lookup
                    if ut.is_method(root_func) and spec.args[0] == attr:
                        subdict = root_func.im_class.__dict__
                    else:
                        # FIXME TODO lookup_attribute_chain
                        subdict = hack_lookup_mod_attrs(attr)
                        if subdict is None:
                            print('Unable to find attribute of attr=%r' % (attr,))
                            if ut.SUPER_STRICT:
                                raise
        if subdict is not None:
            subfunc = subdict[subtup[-1]]
        return subfunc

    def check_subfunc_name(subfunc_name):
        if isinstance(subfunc_name, tuple) or '.' in subfunc_name:
            subfunc = resolve_attr_subfunc(subfunc_name)
        else:
            # try to directly take func from globals
            func_globals = meta_util_six.get_funcglobals(root_func)
            try:
                subfunc = func_globals[subfunc_name]
            except KeyError:
                print('Unable to find function definition subfunc_name=%r' % (subfunc_name,))
                if ut.SUPER_STRICT:
                    raise
                subfunc = None
        if subfunc is not None:
            subkw_list = recursive_parse_kwargs(subfunc)
            have_keys = set(ut.get_list_column(kwargs_list, 0))
            new_subkw = [item for item in subkw_list if item[0] not in have_keys]
        else:
            new_subkw = []
        return new_subkw

    if spec.keywords is not None:
        if VERBOSE_INSPECT:
            print('[inspect] Checking spec.keywords=%r' % (spec.keywords,))
        subfunc_name_list = ut.find_child_kwarg_funcs(sourcecode, spec.keywords)
        if VERBOSE_INSPECT:
            print('[inspect] Checking subfunc_name_list=%r' % (subfunc_name_list,))
        for subfunc_name in subfunc_name_list:
            new_subkw = check_subfunc_name(subfunc_name)
            kwargs_list.extend(new_subkw)
    return kwargs_list


def parse_func_kwarg_keys(func, with_vals=False):
    """ hacky inference of kwargs keys """
    sourcecode = get_func_sourcecode(func, strip_docstr=True,
                                        strip_comments=True)
    kwkeys = parse_kwarg_keys(sourcecode, with_vals=with_vals)
    return kwkeys


def get_func_kwargs(func, stripdef=False, stripret=False, strip_docstr=False, remove_linenums=None):
    """
    func = ibeis.run_experiment
    """
    import utool as ut
    argspec = ut.get_func_argspec(func)
    header_kw = dict(zip(argspec.args[::-1], argspec.defaults[::-1]))
    # TODO
    if argspec.keywords is not None:
        # parse our keywords from func body if possible
        # possibly recursively
        pass
    return header_kw


def parse_kwarg_keys(source, keywords='kwargs', with_vals=False):
    r""" very hacky way to infer some of the kwarg keys

    TODO: use a code parse tree here.  Use hints.  Find other docstrings of
    functions that are called with kwargs. Find the name of the kwargs
    variable.

    Args:
        source (str):

    Returns:
        list: kwarg_keys

    CommandLine:
        python -m utool.util_inspect --exec-parse_kwarg_keys

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> source = "\n  kwargs.get('foo', None)\n  kwargs.pop('bar', 3)\n kwargs.pop('str', '3fd')\n kwargs.pop('str', '3f\'d')\n  \"kwargs.get('baz', None)\""
        >>> print(source)
        >>> kwarg_keys = parse_kwarg_keys(source, with_vals=True)
        >>> result = ('kwarg_keys = %s' % (str(kwarg_keys),))
        >>> print(result)

    """
    #source = ut.get_func_sourcecode(func, strip_docstr=True, strip_comments=True)
    import re
    import utool as ut
    keyname = ut.named_field('keyname', ut.REGEX_VARNAME)
    esc = re.escape
    #default = ut.named_field('default', '[\'\"A-Za-z_][A-Za-z0-9_\'\"]*')
    itemgetter = ut.regex_or(['get', 'pop'])
    pattern = esc(keywords + '.') + itemgetter + esc("('") + keyname + esc("',")
    if with_vals:
        WS = ut.REGEX_WHITESPACE
        valname = WS + ut.named_field('valname', ut.REGEX_RVAL) + WS + esc(')')
        pattern += valname
    #not_quotes = '^' + ut.positive_lookbehind(r'[^\'\"]*')
    #not_quotes = ut.regex_or(['^', r'\n']) + r'[^\'\"]*'
    #not_quotes = r'[^\'\"]*'
    not_quotes = r'^[^\'\"]*'
    pattern = not_quotes + pattern
    regex = re.compile(pattern, flags=re.MULTILINE)
    #print('pattern = %s' % (pattern,))
    #print(pattern)
    groupdict_list = [match.groupdict() for match in regex.finditer(source)]
    kwarg_keys = [groupdict_['keyname'] for groupdict_ in groupdict_list]
    if with_vals:
        kwarg_vals = [ut.smart_cast2(groupdict_['valname']) for groupdict_ in groupdict_list]
        return list(zip(kwarg_keys, kwarg_vals))
    else:
        return kwarg_keys


class KWReg(object):
    """
    Helper to register keywords for complex keyword parsers
    """
    def __init__(kwreg, enabled=False):
        kwreg.keys = []
        kwreg.defaults = []
        kwreg.enabled = enabled

    def __call__(kwreg, key, default):
        if kwreg.enabled:
            kwreg.keys.append(key)
            kwreg.defaults.append(default)
        return key, default

    @property
    def defaultkw(kwreg):
        return dict(zip(kwreg.keys, kwreg.defaults))

    def print_defaultkw(kwreg):
        print(ut.dict_str(kwreg.defaultkw))


def infer_function_info(func):
    r"""
    Infers information for make_default_docstr
    # TODO: Interleave old documentation with new documentation

    Args:
        func (function): live python function

    CommandLine:
        python -m utool --tf infer_function_info:0
        python -m utool --tf infer_function_info:1 --funcname=ibeis_cnn.models.siam.ignore_hardest_cases

    Ignore:
        import ibeis
        func = ibeis.control.IBEISControl.IBEISController.query_chips

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> func = ut.infer_function_info
        >>> #func = ut.Timer.tic
        >>> func = ut.make_default_docstr
        >>> funcinfo = infer_function_info(func)
        >>> result = ut.dict_str(funcinfo.__dict__)
        >>> print(result)

    Example1:
        >>> # SCRIPT
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> funcname = ut.get_argval('--funcname')
        >>> # Parse out custom function
        >>> modname = '.'.join(funcname.split('.')[0:-1])
        >>> script = 'import {modname}\nfunc = {funcname}'.format(
        >>>     modname=modname, funcname=funcname)
        >>> globals_, locals_ = {}, {}
        >>> exec(script, globals_, locals_)
        >>> func = locals_['func']
        >>> funcinfo = infer_function_info(func)
        >>> result = ut.dict_str(funcinfo.__dict__)
        >>> print(result)
    """
    import utool as ut
    import re

    if isinstance(func, property):
        func = func.fget
    try:
        doc_shortdesc = ''
        doc_longdesc = ''

        known_arginfo = ut.ddict(dict)

        if True:
            current_doc = inspect.getdoc(func)
            docstr_blocks = ut.parse_docblocks_from_docstr(current_doc)
            docblock_types = ut.get_list_column(docstr_blocks, 0)
            docblock_types = [re.sub('Example[0-9]', 'Example', type_)
                              for type_ in docblock_types]
            docblock_dict = ut.group_items(docstr_blocks, docblock_types)

            if '' in docblock_dict:
                docheaders = docblock_dict['']
                docheaders_lines = ut.get_list_column(docheaders, 1)
                docheaders_order = ut.get_list_column(docheaders, 2)
                docheaders_lines = ut.sortedby(docheaders_lines, docheaders_order)
                doc_shortdesc = '\n'.join(docheaders_lines)

            if 'Args' in docblock_dict:
                argblocks = docblock_dict['Args']
                if len(argblocks) != 1:
                    print('Warning: should only be one args block')
                else:
                    argblock = argblocks[0][1]

                    assert argblock.startswith('Args:\n')
                    argsblock_ = argblock[len('Args:\n'):]
                    arglines = re.split(r'^    \b', argsblock_, flags=re.MULTILINE)
                    arglines = [line for line in arglines if len(line) > 0]

                    esc = re.escape

                    def escparen(pat):
                        return esc('(')  + pat + esc(')')
                    argname = ut.named_field('argname', ut.REGEX_VARNAME)
                    argtype_ = ut.named_field('argtype', '.' + ut.REGEX_NONGREEDY)
                    argtype = escparen(argtype_)
                    argdesc = ut.named_field('argdesc', '.*')
                    WS = ut.REGEX_WHITESPACE
                    argpattern = (
                        WS + argname + WS + argtype + WS + ':' + WS + argdesc)

                    for argline in arglines:
                        m = re.match(argpattern, argline, flags=re.MULTILINE | re.DOTALL)
                        try:
                            groupdict_ = m.groupdict()
                        except Exception as ex:
                            print('---')
                            print('argline = \n%s' % (argline,))
                            print('---')
                            raise Exception('Unable to parse argline=%s' % (argline,))
                        #print('groupdict_ = %s' % (ut.dict_str(groupdict_),))
                        argname = groupdict_['argname']
                        known_arginfo[argname]['argdesc'] = groupdict_['argdesc'].rstrip('\n')
                        # TODO: record these in a file for future reference
                        # and potential guessing
                        if groupdict_['argtype'] != '?':
                            known_arginfo[argname]['argtype'] = groupdict_['argtype']

        is_class = isinstance(func, six.class_types)

        needs_surround = current_doc is None or len(current_doc) == 0

        if is_class:
            argfunc = func.__init__
        else:
            argfunc = func
        argspec = ut.get_func_argspec(argfunc)
        (argname_list, varargs, varkw, defaults) = argspec

        # See util_inspect
        tup = ut.infer_arg_types_and_descriptions(argname_list, defaults)
        argtype_list, argdesc_list, argdefault_list, hasdefault_list = tup
        # Put in user parsed info
        for index, argname in enumerate(argname_list):
            if argname in known_arginfo:
                arginfo = known_arginfo[argname]
                if 'argdesc' in arginfo:
                    argdesc_list[index] = arginfo['argdesc']
                if 'argtype' in arginfo:
                    argtype_list[index] = arginfo['argtype']

        if not is_class:
            # Move source down to base indentation, but remember original indentation
            sourcecode = get_func_sourcecode(func)
            #kwarg_keys = ut.parse_kwarg_keys(sourcecode)
            kwarg_items = ut.recursive_parse_kwargs(func)

            kwarg_keys = ut.get_list_column(kwarg_items, 0)
            #kwarg_keys = ut.unique_ordered(kwarg_keys)
            kwarg_keys = ut.setdiff_ordered(kwarg_keys, argname_list)
        else:
            sourcecode = None
            kwarg_keys = []

        if sourcecode is not None:
            num_indent = ut.get_indentation(sourcecode)
            sourcecode = ut.unindent(sourcecode)
            returninfo = ut.parse_return_type(sourcecode)
        else:
            num_indent = 0
            returninfo = None, None, None, ''
        return_type, return_name, return_header, return_desc = returninfo

        modname = func.__module__
        funcname = ut.get_funcname(func)
    except Exception as ex:
        #print('dealing with infer function error')
        #print('has utinfo? ' + str(hasattr(func, '_utinfo')))
        #sourcefile = inspect.getsourcefile(func)  # NOQA
        ut.printex(ex, 'Error Infering Function Info', keys=[
            'func',
            'sourcefile',
            'sourcecode',
            'argspec',
        ])
        raise

    class FunctionInfo(object):
        def __init__(self):
            pass
    funcinfo = FunctionInfo()
    funcinfo.needs_surround = needs_surround
    funcinfo.argname_list = argname_list
    funcinfo.argtype_list = argtype_list
    funcinfo.argdesc_list = argdesc_list
    funcinfo.argdefault_list = argdefault_list
    funcinfo.hasdefault_list = hasdefault_list
    funcinfo.kwarg_keys = kwarg_keys
    funcinfo.varargs = varargs
    funcinfo.varkw = varkw
    funcinfo.defaults = defaults
    funcinfo.num_indent = num_indent
    funcinfo.return_type = return_type
    funcinfo.return_name = return_name
    funcinfo.return_header = return_header
    funcinfo.return_desc = return_desc
    funcinfo.modname = modname
    funcinfo.funcname = funcname
    funcinfo.doc_shortdesc = doc_shortdesc
    funcinfo.doc_longdesc = doc_longdesc
    funcinfo.ismethod = hasattr(func, 'im_class')
    return funcinfo


def parse_callname(searchline, sentinal='def '):
    """
    Parses the function or class name from a signature line
    originally part of the vim plugin
    """
    rparen_pos = searchline.find('(')
    if rparen_pos > 0:
        callname = searchline[len(sentinal):rparen_pos].strip(' ')
        return callname
    return None


def find_pattern_above_row(pattern, line_list, row, maxIter=50):
    """ originally part of the vim plugin """
    import re
    # Janky way to find function name
    ix = 0
    while True:
        pos = row - ix
        if maxIter is not None and ix > maxIter:
            break
        if pos < 0:
            break
            raise AssertionError('end of buffer')
        searchline = line_list[pos]
        if re.match(pattern, searchline) is not None:
            return searchline, pos
        ix += 1


def find_pyclass_above_row(line_list, row):
    """ originally part of the vim plugin """
    # Get text posision
    pattern = '^class [a-zA-Z_]'
    classline, classpos = find_pattern_above_row(pattern, line_list, row, maxIter=None)
    return classline, classpos


def find_pyfunc_above_row(line_list, row, orclass=False):
    """
    originally part of the vim plugin

    CommandLine:
        python -m utool.util_inspect --test-find_pyfunc_above_row

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> func = find_pyfunc_above_row
        >>> fpath = meta_util_six.get_funcglobals(func)['__file__'].replace('.pyc', '.py')
        >>> line_list = ut.read_from(fpath, aslines=True)
        >>> row = meta_util_six.get_funccode(func).co_firstlineno + 1
        >>> pyfunc, searchline = find_pyfunc_above_row(line_list, row)
        >>> result = pyfunc
        >>> print(result)
        find_pyfunc_above_row

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> fpath = ut.util_inspect.__file__.replace('.pyc', '.py')
        >>> line_list = ut.read_from(fpath, aslines=True)
        >>> row = 1608
        >>> pyfunc, searchline = find_pyfunc_above_row(line_list, row, orclass=True)
        >>> result = pyfunc
        >>> print(result)
        find_pyfunc_above_row
    """
    searchlines = []  # for debugging
    funcname = None
    # Janky way to find function name
    func_sentinal   = 'def '
    method_sentinal = '    def '
    class_sentinal = 'class '
    for ix in range(200):
        func_pos = row - ix
        searchline = line_list[func_pos]
        cleanline = searchline.strip(' ')
        searchlines.append(cleanline)
        if searchline.startswith(func_sentinal):  # and cleanline.endswith(':'):
            # Found a valid function name
            funcname = parse_callname(searchline, func_sentinal)
            if funcname is not None:
                break
        if orclass and searchline.startswith(class_sentinal):
            # Found a valid class name (as funcname)
            funcname = parse_callname(searchline, class_sentinal)
            if funcname is not None:
                break
        if searchline.startswith(method_sentinal):  # and cleanline.endswith(':'):
            # Found a valid function name
            funcname = parse_callname(searchline, method_sentinal)
            if funcname is not None:
                classline, classpos = find_pyclass_above_row(line_list, func_pos)
                classname = parse_callname(classline, class_sentinal)
                if classname is not None:
                    funcname = '.'.join([classname, funcname])
                    break
                else:
                    funcname = None
    return funcname, searchlines


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_inspect; utool.doctest_funcs(utool.util_inspect, allexamples=True)"
        python -c "import utool, utool.util_inspect; utool.doctest_funcs(utool.util_inspect)"
        python -m utool.util_inspect --enableall
        python -m utool.util_inspect --enableall --test-iter-module-doctestable:1
        python -m utool.util_inspect --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
