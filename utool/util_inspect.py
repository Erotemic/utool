from __future__ import absolute_import, division, print_function
import inspect
import types
import six
import functools
from utool import util_regex
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
            #if ut.VERBOSE:
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
        >>> (return_type, return_name, return_header) = returninfo
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
    return_name = None
    return_type = '?'
    if returnnode is None:
        return_type = 'None'
    elif isinstance(returnnode.value, ast.Tuple):
        names = returnnode.value.elts
        tupleid = '(%s)' % (', '.join([str(name.id) for name in names]))
        return_type = 'tuple'
        return_name = tupleid
    elif isinstance(returnnode.value, ast.Dict):
        return_type = 'dict'
        return_name = None
    elif isinstance(returnnode.value, ast.Name):
        return_name = returnnode.value.id
    else:
        return_type = str(type(returnnode.value))
    return return_type, return_name, return_header


def get_func_sourcecode(func):
    """
    wrapper around inspect.getsource but takes into account utool decorators
    """
    sourcefile = inspect.getsourcefile(func)
    if sourcefile is not None:
        return inspect.getsource(func)
    else:
        if hasattr(func, '_utinfo'):
            return func._utinfo['src']
            return get_func_sourcecode(func._utinfo['orig_func'])
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
        returninfo = None, None, None
    return_type, return_name, return_header = returninfo

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
    funcinfo.modname = modname
    funcinfo.funcname = funcname

    return funcinfo


def infer_arg_types_and_descriptions(argname_list, defaults):
    """
    Args:
        argname_list (list):
        defaults (?):

    Returns:
        tuple : (arg_types, argdesc_list)

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> argname_list = ['ibs', 'qaid', 'fdKfds']
        >>> defaults = None
        >>> arg_types, argdesc_list = utool.infer_arg_types_and_descriptions(argname_list, defaults)
    """
    import utool as ut

    # hacks for IBEIS
    if ut.is_developer():
        # key = regex pattern
        # val = hint=tuple(type_, desc_)
        from collections import OrderedDict
        registered_hints = OrderedDict([
            ('ibs.*'   , ('IBEISController', None)),
            ('qreq_'   , ('QueryRequest', 'hyper-parameters')),
            ('qres.*'  , ('QueryResult', 'object of feature correspondences and scores')),
            ('qparams*', ('QueryParams', 'hyper-parameters')),
            ('K'       , ('int', None)),
            ('Knorm'   , ('int', None)),
            ('smk_alpha',  ('float', 'selectivity power')),
            ('smk_thresh', ('float', 'selectivity threshold')),
            ('query_sccw', ('float', 'query self-consistency-criterion')),
            ('data_sccw', ('float', 'data self-consistency-criterion')),
            ('invindex', ('InvertedIndex', 'object for fast vocab lookup')),
            ('vecs'    , ('ndarray', None)),
            ('maws'    , ('ndarray', None)),
            ('words'   , ('ndarray', None)),
            ('word'    , ('ndarray', None)),
            ('rvecs'   , ('ndarray', None)),
            ('wx2_'    , ('dict', None)),
            ('qfx2_.*' , ('ndarray', None)),
            ('.+2_.*'  , ('dict', None)),
            ('.*_list' , ('list', None)),
            ('.*_sublist' , ('list', None)),
            ('qaid'    , ('int', 'query annotation id')),
            ('qnid'    , ('int', 'query name id')),
        ])

    if defaults is None:
        defaults = []
    default_types = [type(val).__name__.replace('NoneType', 'None') for val in defaults]
    arg_types = ['?'] * (len(argname_list) - len(defaults)) + default_types

    argdesc_list = ['' for _ in range(len(argname_list))]

    # use hints to build better docstrs
    for argx in range(len(argname_list)):
        if arg_types[argx] == '?' or arg_types[argx] == 'None':
            argname = argname_list[argx]
            for regex, hint in six.iteritems(registered_hints):
                if util_regex.regex_matches(regex, argname):
                    type_ = hint[0]
                    desc_ = hint[1]
                    if type_ is not None:
                        arg_types[argx] = type_
                    if desc_ is not None:
                        argdesc_list[argx] = ' ' + desc_
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
