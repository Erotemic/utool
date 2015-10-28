from __future__ import absolute_import, division, print_function
import inspect
import types
import six
import re
import functools
from collections import OrderedDict
from six.moves import range, zip  # NOQA
from utool import util_regex
from utool import util_arg
from utool import util_inject
from utool._internal import meta_util_six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[inspect]')


def get_dev_hints():
    VAL_FIELD = util_regex.named_field('val', '.*')
    VAL_BREF = util_regex.bref_field('val')

    registered_hints = OrderedDict([
        # General IBEIS hints
        ('ibs.*'   , ('IBEISController', 'ibeis controller object')),
        ('test_result', ('TestResult', 'test result object')),
        ('qaid2_qres.*'   , ('dict', 'dict of query result objects')),
        ('qreq_'   , ('QueryRequest', 'query request object with hyper-parameters')),
        ('qres.*'  , ('QueryResult', 'object of feature correspondences and scores')),
        ('qparams*', ('QueryParams', 'query hyper-parameters')),
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
        ('_path' , ('str', 'path string')),
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
        ('.+2_.*'  , ('dict', None)),
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
        python -c "import utool; print(utool.auto_docstr('ibeis.model.hots.pipeline', 'build_chipmatches'))"

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
            import types
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
                            include_methods=True, include_inherited=False):
    r"""
    Yeilds doctestable live object form a modules

    Args:
        module (live python module):
        include_funcs (bool):
        include_classes (bool):
        include_methods (bool):

    Yeilds:
        tuple (str, callable): (funcname, func) doctestable

    CommandLine:
        python -m utool.util_inspect --test-iter_module_doctestable --modname=ibeis.model.hots.chip_match
        python -m utool.util_inspect --test-iter_module_doctestable --modname=ibeis.control.IBEISControl
        python -m utool.util_inspect --test-iter_module_doctestable --modname=ibeis.control.SQLDatabaseControl

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *   # NOQA
        >>> import utool as ut
        >>> modname = ut.get_argval('--modname', type_=str, default=None)
        >>> module = ut.util_tests if modname is None else ut.import_modname(modname)
        >>> doctestable_list = list(iter_module_doctestable(module))
        >>> func_names = sorted(ut.get_list_column(doctestable_list, 0))
        >>> print('\n'.join(func_names))
    """
    import ctypes
    valid_func_types = (types.FunctionType, types.BuiltinFunctionType, classmethod,
                        #types.MethodType, types.BuiltinMethodType,
                        )
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

    #modpath = ut.get_modname_from_modpath(module.__file__)

    for key, val in six.iteritems(module.__dict__):
        if hasattr(val, '__module__'):
            # HACK: todo. figure out true parent module
            if val.__module__ == 'numpy':
                continue
            #if key == 'NP_NDARRAY':
            #    import utool as ut
            #    ut.embed()

        if val is None:
            pass
        elif isinstance(val, valid_func_types):
            if include_funcs:
                yield key, val
        elif isinstance(val, valid_class_types):
            class_ = val
            if include_classes:
                yield key, val
            if include_methods:
                for subkey, subval in six.iteritems(class_.__dict__):
                    #if subkey.startswith('get_a'):
                    #    import utool as ut
                    #    ut.embed()
                    # Unbound methods are still typed as functions
                    if isinstance(subval, valid_func_types):
                        if hasattr(subval, 'func_globals') and subval.func_globals['__name__'] != module.__name__:
                            if not include_inherited:
                                continue
                        if not isinstance(subval, types.BuiltinFunctionType) and not isinstance(subval, classmethod):
                            # HACK: __ut_parent_class__ lets util_test have
                            # more info on the func should return extra info
                            # instead
                            subval.__ut_parent_class__ = class_
                        yield subkey, subval
                    elif isinstance(val, invalid_types):
                        pass
                    else:
                        #import utool as ut
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


# grep is in util_path. Thats pretty inspecty

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
        keys = ut.list_take(argspec.args, range(num_args - num_keys, num_args))
    else:
        keys = []
    is_arbitrary = argspec.keywords is not None
    RECURSIVE = False
    if RECURSIVE and argspec.keywords is not None:
        pass
        # TODO: look inside function at the functions that the kwargs object is being
        # passed to
    return keys, is_arbitrary


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
    debug = False

    if debug:
        print('\nInput:')
        print('target_kwargs_name = %r' % (target_kwargs_name,))
        print('\nSource:')
        print(sourcecode)
        import astor
        print('\nParse:')
        print(astor.dump(pt))

    class CallVisitor(ast.NodeVisitor):
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
    CallVisitor().visit(pt)
    return child_funcnamess
    #print('child_funcnamess = %r' % (child_funcnamess,))


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
        >>> (return_type, return_name, return_header, return_desc) = returninfo
        >>> result = str((return_type, return_name, return_header))
        >>> print(result)
        ('?', 'bar', 'Returns')
    """
    import utool as ut
    import ast
    if six.PY3:
        Try = ast.Try
    else:
        Try = ast.TryExcept
    if ut.VERBOSE:
        print('[utool] parsing return types')

    if sourcecode is None:
        return_type, return_name, return_header = (None, None, None)
        return return_type, return_name, return_header, None

    #source_lines = sourcecode.splitlines()
    sourcecode = 'from __future__ import print_function\n' + sourcecode
    try:
        pt = ast.parse(sourcecode)
    except Exception:
        return_type, return_name, return_header = (None, None, None)
        raise
        return return_type, return_name, return_header, None
        #print(sourcecode)
        #ut.printex(ex, 'Error Parsing')

    assert isinstance(pt, ast.Module), str(type(pt))

    def find_function_nodes(pt):
        function_nodes = []
        for node in pt.body:
            if isinstance(node, ast.FunctionDef):
                function_nodes.append(node)
        return function_nodes

    function_nodes = find_function_nodes(pt)
    assert len(function_nodes) == 1
    func_node = function_nodes[0]

    def find_return_node(node):
        if isinstance(node, list):
            candidates = []
            node_list = node
            for node in node_list:
                candidate = find_return_node(node)
                if candidate is not None:
                    candidates.append(candidate)
            if len(candidates) > 0:
                return candidates[0]
        elif isinstance(node, (ast.Return, ast.Yield)):
            return node
        elif isinstance(node, (ast.If, Try)):
            return find_return_node(node.body)
        else:
            pass
            #print(type(node))
    if ut.VERBOSE:
        print('[utool] parsing return types')
    returnnode = find_return_node(func_node.body)
    # Check return or yeild
    if isinstance(returnnode, ast.Yield):
        return_header = 'Yeilds'
    elif isinstance(returnnode, ast.Return):
        return_header = 'Returns'
    else:
        return_header = None
    # Get more return info

    def get_node_name_and_type(node):
        node_name = None
        node_type = '?'
        if node is None:
            node_type = 'None'
        elif isinstance(node.value, ast.Tuple):
            tupnode_list = node.value.elts
            def get_tuple_membername(tupnode):
                if hasattr(tupnode, 'id'):
                    return tupnode.id
                #elif hasattr(tupnode, 'key'):
                #    return 'tupnode.key=%r' % (tupnode.key,)
                elif hasattr(tupnode, 'value'):
                    return 'None'
                    #return 'tupnode.value=%r' % (tupnode.value,)
                    #return (ast.dump(tupnode))
                    #return get_node_name_and_type(tupnode)[1]
                #type(name.value)
                else:
                    return 'None'
                pass
            tupleid = '(%s)' % (', '.join([str(get_tuple_membername(tupnode)) for tupnode in tupnode_list]))
            node_type = 'tuple'
            node_name = tupleid
            #node_name = ast.dump(node)
        elif isinstance(node.value, ast.Dict):
            node_type = 'dict'
            node_name = None
        elif isinstance(node.value, ast.Name):
            node_name = node.value.id
        else:
            #node_type = 'ADD_TO_GET_NODE_NAME_AND_TYPE: ' + str(type(node.value))
            node_type = '?'
        return node_type, node_name

    return_type, return_name = get_node_name_and_type(returnnode)

    if return_type == '?':
        argtype_list, argdesc_list, argdefault_list, hasdefault_list = infer_arg_types_and_descriptions([return_name], [])
        return_type = argtype_list[0]
        return_desc = argdesc_list[0]
    else:
        return_desc = ''

    return return_type, return_name, return_header, return_desc


def exec_func_sourcecode(func, globals_, locals_, key_list):
    """ execs a func and returns requested local vars """
    import utool as ut
    sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True)
    six.exec_(sourcecode, globals_, locals_)
    # Draw intermediate steps
    if key_list is None:
        return locals_
    else:
        var_list = ut.dict_take( locals_, key_list)
        return var_list

exec_func_src = exec_func_sourcecode


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


def get_func_sourcecode(func, stripdef=False, stripret=False, strip_docstr=False, strip_comments=False, remove_linenums=None):
    """
    wrapper around inspect.getsource but takes into account utool decorators
    strip flags are very hacky as of now

    Args:
        func (?):
        stripdef (bool):


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
        try:
            #print(func)
            sourcecode = inspect.getsource(func)
            #print(sourcecode)
        except OSError as ex:
            ut.printex(ex, 'Error getting source', keys=['sourcefile'])
            raise
    else:
        sourcecode = None
    #orig_source = sourcecode
    #print(orig_source)
    if stripdef:
        # hacky
        sourcecode = ut.unindent(sourcecode)
        #sourcecode = ut.unindent(ut.regex_replace('def [^)]*\\):\n', '', sourcecode))
        sourcecode = ut.unindent(ut.regex_replace('def [^:]*\\):\n', '', sourcecode))
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


def get_func_argspec(func):
    """
    wrapper around inspect.getargspec but takes into account utool decorators
    """
    if hasattr(func, '_utinfo'):
        argspec = func._utinfo['orig_argspec']
        return argspec
    argspec = inspect.getargspec(func)
    return argspec


def recursive_parse_kwargs(root_func, path_=None):
    """ recursive kwargs parser """
    import utool as ut
    if path_ is None:
        path_ = []
    if root_func in path_:
        return []
    path_.append(root_func)
    spec = ut.get_func_argspec(root_func)
    # ADD MORE
    kwargs_list = ut.get_kwargs(root_func)[0]
    if spec.keywords is not None:
        sourcecode = ut.get_func_sourcecode(root_func, strip_docstr=True,
                                            stripdef=True)
        subfunc_name_list = ut.find_child_kwarg_funcs(sourcecode, spec.keywords)
        for subfunc_name in subfunc_name_list:
            subfunc = root_func.func_globals[subfunc_name]
            kwargs_list.extend(recursive_parse_kwargs(subfunc))
    return kwargs_list


def parse_func_kwarg_keys(func, with_vals=False):
    """ hacky inference of kwargs keys """
    sourcecode = get_func_sourcecode(func, strip_docstr=True,
                                        strip_comments=True)
    kwkeys = parse_kwarg_keys(sourcecode, with_vals=with_vals)
    return kwkeys


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

    Args:
        func (function): live python function

    CommandLine:
        python -m utool.util_inspect --test-infer_function_info:0
        python -m utool.util_inspect --exec-infer_function_info:1 --funcname=ibeis_cnn.models.siam.ignore_hardest_cases --exec-mode

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> #func = ut.infer_function_info
        >>> #func = ut.Timer.tic
        >>> func = get_func_sourcecode
        >>> funcinfo = infer_function_info(func)
        >>> result = ut.dict_str(funcinfo.__dict__)
        >>> print(result)

    Example:
        >>> # SCRIPT
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> funcname = ut.get_argval('--funcname')
        >>> # Parse out custom function
        >>> modname = '.'.join(funcname.split('.')[0:-1])
        >>> script = 'import {modname}\nfunc = {funcname}'.format(modname=modname, funcname=funcname)
        >>> globals_, locals_ = {}, {}
        >>> exec(script, globals_, locals_)
        >>> func = locals_['func']
        >>> funcinfo = infer_function_info(func)
        >>> result = ut.dict_str(funcinfo.__dict__)
        >>> print(result)
    """
    import utool as ut
    try:
        current_doc = inspect.getdoc(func)
        needs_surround = current_doc is None or len(current_doc) == 0
        argspec = ut.get_func_argspec(func)
        (argname_list, varargs, varkw, defaults) = argspec

        # See util_inspect
        argtype_list, argdesc_list, argdefault_list, hasdefault_list = ut.infer_arg_types_and_descriptions(argname_list, defaults)

        # Move source down to base indentation, but remember original indentation
        sourcecode = get_func_sourcecode(func)
        kwarg_keys = ut.parse_kwarg_keys(sourcecode)
        num_indent = ut.get_indentation(sourcecode)
        sourcecode = ut.unindent(sourcecode)

        if sourcecode is not None:
            returninfo = ut.parse_return_type(sourcecode)
        else:
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


def find_pyfunc_above_row(line_list, row):
    """
    originally part of the vim plugin

    CommandLine:
        python -m utool.util_inspect --test-find_pyfunc_above_row

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> #fpath = ut.truepath('~/code/ibeis/ibeis/control/IBEISControl.py')
        >>> #fpath = ut.truepath('~/code/utool/utool/util_inspect.py')
        >>> fpath = ut.truepath('~/code/ibeis_cnn/ibeis_cnn/models.py')
        >>> line_list = ut.read_from(fpath, aslines=True)
        >>> #row = 200
        >>> row = 93
        >>> pyfunc, searchline = find_pyfunc_above_row(line_list, row)
        >>> print(pyfunc)
    """
    searchlines = []  # for debugging
    funcname = None
    # Janky way to find function name
    func_sentinal   = 'def '
    method_sentinal = '    def '
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
        if searchline.startswith(method_sentinal):  # and cleanline.endswith(':'):
            # Found a valid function name
            funcname = parse_callname(searchline, method_sentinal)
            if funcname is not None:
                classline, classpos = find_pyclass_above_row(line_list, func_pos)
                classname = parse_callname(classline, 'class ')
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
