from __future__ import absolute_import, division, print_function
import inspect
import types
import six
import re
import functools
from utool import util_regex
from utool import util_arg
from utool import util_inject
from utool._internal import meta_util_six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[inspect]')


def iter_module_doctestable(module, include_funcs=True, include_classes=True,
                            include_methods=True):
    r"""
    Yeilds doctestable live object form a modules

    Args:
        module (live python module):
        include_funcs (bool):
        include_classes (bool):
        include_methods (bool):

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *   # NOQA
        >>> import utool as ut
        >>> module = ut.util_tests
        >>> doctestable_list = list(iter_module_doctestable(module))
        >>> func_names = ut.get_list_column(doctestable_list, 0)
        >>> print('\n'.join(func_names))

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *   # NOQA
        >>> import utool as ut
        >>> import ibeis
        >>> import ibeis.control.IBEISControl
        >>> module = ibeis.control.IBEISControl
        >>> doctestable_list = list(iter_module_doctestable(module))
        >>> func_names = ut.get_list_column(doctestable_list, 0)
        >>> print('\n'.join(func_names))
    """
    valid_func_types = (types.FunctionType, types.BuiltinFunctionType,
                        #types.MethodType, types.BuiltinMethodType,
                        )
    valid_class_types = (types.ClassType,  types.TypeType,)

    scalar_types = ([dict, list, tuple, set, frozenset, bool, float, int] +
                    list(six.string_types))
    scalar_types += list(six.string_types)
    other_types = [types.InstanceType, functools.partial, types.ModuleType]
    invalid_types = tuple(scalar_types + other_types)

    for key, val in six.iteritems(module.__dict__):
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
                    # Unbound methods are still typed as functions
                    if isinstance(subval, valid_func_types):
                        yield subkey, subval
        elif isinstance(val, invalid_types):
            pass
        else:
            #import utool as ut
            if util_arg.VERBOSE:
                print('[util_inspect] WARNING:')
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
        >>> import utool
        >>> sourcecode = utool.codeblock(
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

    import utool
    import ast
    if utool.VERBOSE:
        print('[utool] parsing return types')

    if sourcecode is None:
        return_type, return_name, return_header = (None, None, None)
        return return_type, return_name, return_header

    #source_lines = sourcecode.splitlines()
    sourcecode = 'from __future__ import print_function\n' + sourcecode
    pt = ast.parse(sourcecode)

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
        elif isinstance(node, (ast.If, ast.TryExcept)):
            return find_return_node(node.body)
        else:
            pass
            #print(type(node))
    if utool.VERBOSE:
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
        arg_types, arg_desc = infer_arg_types_and_descriptions([return_name], [])
        return_type = arg_types[0]
        return_desc = arg_desc[0]
    else:
        return_desc = ''

    return return_type, return_name, return_header, return_desc


def get_func_sourcecode(func):
    """
    wrapper around inspect.getsource but takes into account utool decorators
    """
    sourcefile = inspect.getsourcefile(func)
    if hasattr(func, '_utinfo'):
        #return func._utinfo['src']
        return get_func_sourcecode(func._utinfo['orig_func'])
    if sourcefile is not None:
        return inspect.getsource(func)
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


def infer_function_info(func):
    """
    Infers information for make_default_docstr

    Args:
        func (function): live python function

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> func = ut.infer_function_info
        >>> func = ut.Timer.tic
        >>> funcinfo = infer_function_info(func)
        >>> result = ut.dict_str(funcinfo.__dict__)
        >>> print(result)
    """
    import utool as ut
    current_doc = inspect.getdoc(func)
    needs_surround = current_doc is None or len(current_doc) == 0
    argspec = ut.get_func_argspec(func)
    (argname_list, varargs, varkw, defaults) = argspec

    # See util_inspect
    argtype_list, argdesc_list = ut.infer_arg_types_and_descriptions(argname_list, defaults)

    # Move source down to base indentation, but remember original indentation
    sourcecode = get_func_sourcecode(func)
    num_indent = ut.get_indentation(sourcecode)
    sourcecode = ut.unindent(sourcecode)

    if sourcecode is not None:
        returninfo = ut.parse_return_type(sourcecode)
    else:
        returninfo = None, None, None, ''
    return_type, return_name, return_header, return_desc = returninfo

    modname = func.__module__
    funcname = ut.get_funcname(func)

    class FunctionInfo(object):
        def __init__(self):
            pass
    funcinfo = FunctionInfo()
    funcinfo.needs_surround = needs_surround
    funcinfo.argname_list = argname_list
    funcinfo.argtype_list = argtype_list
    funcinfo.argdesc_list = argdesc_list
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


def infer_arg_types_and_descriptions(argname_list, defaults):
    """
    Args:
        argname_list (list):
        defaults (list):

    Returns:
        tuple : (arg_types, argdesc_list)

    CommandLine:
        python -m utool.util_inspect --test-infer_arg_types_and_descriptions

    Ignore:
        python -c "import utool; print(utool.auto_docstr('ibeis.model.hots.pipeline', 'build_chipmatches'))"

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> argname_list = ['ibs', 'qaid', 'fdKfds', 'qfx2_foo']
        >>> defaults = None
        >>> arg_types, argdesc_list = utool.infer_arg_types_and_descriptions(argname_list, defaults)
    """
    #import utool as ut
    from utool import util_dev

    # hacks for IBEIS
    if True or util_dev.is_developer():
        # key = regex pattern
        # val = hint=tuple(type_, desc_)
        from collections import OrderedDict
        VAL_FIELD = util_regex.named_field('val', '.*')
        VAL_BREF = util_regex.bref_field('val')

        registered_hints = OrderedDict([
            # General IBEIS hints
            ('ibs.*'   , ('IBEISController', 'ibeis controller object')),
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
            ('qaid'    , ('int', 'query annotation id')),
            ('daids'   , ('list', 'database annotation ids')),
            ('qaids'   , ('list', 'query annotation ids')),
            ('use_cache', ('bool', 'turns on disk based caching')),
            ('qreq_vsmany_', ('QueryRequest', 'persistant vsmany query request')),
            ('qnid'    , ('int', 'query name id')),

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
            ('[qd]?kpts[0-9]?', ('ndarray[float32_t, ndim=2]', 'keypoints')),
            ('invV_mats2x2', ('ndarray[float32_t, ndim=3]',  'keypoint shapes')),
            ('invVR_mats2x2', ('ndarray[float32_t, ndim=3]', 'keypoint shape and rotations')),
            ('invV_mats', ('ndarray[float32_t, ndim=3]',  'keypoint shapes (possibly translation)')),
            ('invVR_mats', ('ndarray[float32_t, ndim=3]', 'keypoint shape and rotations (possibly translation)')),
            ('img', ('ndarray[uint8_t, ndim=2]', 'image data')),
            ('pnum', ('tuple', 'plot number')),
            ('fnum', ('int', 'figure number')),
            ('title', ('str', '')),

            # utool hints
            ('funcname'    , ('str', 'function name')),
            ('modname'    , ('str', 'module name')),
            ('argname_list'   , ('str', 'list of argument names')),
            ('return_name'    , ('str', 'return variable name')),
            ('examplecode'    , ('str', None)),


            # My coding style hints
            ('wx2_'    , ('dict', None)),
            ('qfx2_' + VAL_FIELD,
             ('ndarray',
              'mapping from query feature index to ' + VAL_BREF)),
            ('.*x2_.*' , ('ndarray', None)),
            ('.+2_.*'  , ('dict', None)),
            ('.*_?list_?' , ('list', None)),
            ('.*_tup' , ('tuple', None)),
            ('.*_sublist' , ('list', None)),
            ('verbose', ('bool', 'verbosity flag')),
        ])

    if defaults is None:
        defaults = []
    default_types = [type(val).__name__.replace('NoneType', 'None') for val in defaults]
    arg_types = ['?'] * (len(argname_list) - len(defaults)) + default_types

    argdesc_list = ['' for _ in range(len(argname_list))]

    # use hints to build better docstrs
    for argx in range(len(argname_list)):
        #if arg_types[argx] == '?' or arg_types[argx] == 'None':
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
                    if arg_types[argx] == '?' or arg_types[argx] == 'None':
                        arg_types[argx] = type_
                if desc_ is not None:
                    desc_ = matchobj.expand(desc_)
                    argdesc_list[argx] = ' ' + desc_
                break
    return arg_types, argdesc_list


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
