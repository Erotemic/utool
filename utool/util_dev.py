from __future__ import absolute_import, division, print_function
import sys
import six
import re
import os
import gc
import warnings
import weakref
import inspect
from collections import OrderedDict
from six.moves import input
from utool import util_progress
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError as ex:
    HAS_NUMPY = False
    pass
from os.path import splitext, exists, join, split, relpath
from utool import util_inject
from utool import util_regex
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[dev]')

if HAS_NUMPY:
    INDEXABLE_TYPES = (list, tuple, np.ndarray)
else:
    INDEXABLE_TYPES = (list, tuple)


def DEPRICATED(func):
    """ deprication decorator """
    warn_msg = 'Depricated call to: %s' % func.__name__

    def __DEP_WRAPPER(*args, **kwargs):
        raise Exception('dep')
        warnings.warn(warn_msg, category=DeprecationWarning)
        #warnings.warn(warn_msg, category=DeprecationWarning)
        return func(*args, **kwargs)
    __DEP_WRAPPER.__name__ = func.__name__
    __DEP_WRAPPER.__doc__ = func.__doc__
    __DEP_WRAPPER.__dict__.update(func.__dict__)
    return __DEP_WRAPPER


#def ensure_vararg_list(varargs):
#    """
#    It is useful to have a function take a list of objects to act upon.
#    But sometimes you want just one. Varargs lets you pass in as many as you
#    want, and it lets you have just one if needbe.
#    But sometimes the function caller explicitly passes in the list. In that
#    case we parse it out
#    """
#    if len(varargs) == 1:
#        if isinstance(varargs[0], INDEXABLE_TYPES):
#            return varargs[0]
#    return varargs


def input_timeout(msg='Waiting for input...', timeout=30):
    """Function does not work quite right yet.

    Args:
        msg (str):
        timeout (int):

    Returns:
        ?: ans

    References:
        http://stackoverflow.com/questions/1335507/keyboard-input-with-timeout-in-python
        http://home.wlu.edu/~levys/software/kbhit.py
        http://stackoverflow.com/questions/3471461/raw-input-and-timeout/3911560#3911560

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dev import *  # NOQA
        >>> msg = 'Waiting for input...'
        >>> timeout = 30
        >>> ans = input_timeout(msg, timeout)
        >>> print(ans)
    """
    import sys
    import select
    import time
    ans = None
    print('You have %d seconds to answer!' % timeout)
    print(msg)
    if sys.platform.startswith('win32'):
        import msvcrt
        start_time = time.time()
        instr = ''
        while True:
            if msvcrt.kbhit():
                chr_ = msvcrt.getche()
                if ord(chr_) == 13:  # enter_key
                    # Accept input
                    ans = instr
                    break
                elif ord(chr_) >= 32:  # space_char
                    # Append to input
                    instr += chr_
            ellapsed = time.time() - start_time
            if ellapsed > timeout:
                ans = None
        print('')  # needed to move to next line
    else:
        rlist, o, e = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ans = sys.stdin.readline().strip()
    return ans


def autofix_codeblock(codeblock, max_line_len=80,
                      aggressive=False,
                      very_aggressive=False,
                      experimental=False):
    r"""
    Uses autopep8 to format a block of code

    Example:
        >>> import utool
        >>> codeblock = utool.codeblock(
            '''
            def func( with , some = 'Problems' ):


             syntax ='Ok'
             but = 'Its very messy'
             if None:
                    # syntax might not be perfect due to being cut off
                    ommiting_this_line_still_works=   True
            ''')
        >>> fixed_codeblock = utool.autofix_codeblock(codeblock)
        >>> print(fixed_codeblock)
    """
    # FIXME idk how to remove the blank line following the function with
    # autopep8. It seems to not be supported by them, but it looks bad.
    import autopep8
    arglist = ['--max-line-length', '80']
    if aggressive:
        arglist.extend(['-a'])
    if very_aggressive:
        arglist.extend(['-a', '-a'])
    if experimental:
        arglist.extend(['--experimental'])
    arglist.extend([''])
    autopep8_options = autopep8.parse_args(arglist)
    fixed_codeblock = autopep8.fix_code(codeblock, options=autopep8_options)
    return fixed_codeblock


def auto_docstr(modname, funcname, verbose=True):
    """
    Args:
        modname (str):
        funcname (str):

    Returns:
        docstr

    Example:
        >>> import utool
        >>> utool.util_dev.rrr()
        >>> #docstr = utool.auto_docstr('ibeis.model.hots.smk.smk_index', 'compute_negentropy_names')
        >>> modname = 'utool.util_dev'
        >>> funcname = 'auto_docstr'
        >>> docstr = utool.util_dev.auto_docstr(modname, funcname)
        >>> print(docstr)
    """
    import utool
    docstr = 'error'
    if isinstance(modname, str):
        module = __import__(modname)
        import imp
        imp.reload(module)
        #try:
        #    func = getattr(module, funcname)
        #    docstr = make_default_docstr(func)
        #    return docstr
        #except Exception as ex1:
        #docstr = 'error ' + str(ex1)
        #if utool.VERBOSE:
        #    print('make_default_docstr is falling back')
        #print(ex)
        #print('modname = '  + modname)
        #print('funcname = ' + funcname)
        try:
            execstr = utool.codeblock(
                '''
                import {modname}
                import imp
                imp.reload({modname})
                import utool
                imp.reload(utool.util_dev)
                docstr = utool.util_dev.make_default_docstr({modname}.{funcname})
                '''
            ).format(**locals())
            exec(execstr)
            #return 'BARFOOO' +  docstr
            return docstr
            #print(execstr)
        except Exception as ex2:
            docstr = 'error ' + str(ex2)
            if verbose:
                import utool
                #utool.printex(ex1, 'ex1')
                utool.printex(ex2, 'ex2', tb=True)
            error_str = utool.formatex(ex2, 'ex2', tb=True)
            return error_str
            #return docstr + '\n' + execstr
    else:
        docstr = 'error'
    return docstr


def strip_line_comments(code_text, comment_char='#'):
    import utool as ut
    comment_regex = comment_char + ' .[^\n]*$'
    # full line comments
    code_text = ut.regex_replace('^ *' + comment_regex + '\n', '', code_text)
    # inline comments
    code_text = ut.regex_replace('  ' + comment_regex, '', code_text)
    return code_text


def print_auto_docstr(modname, funcname):
    """
    python -c "import utool; utool.print_auto_docstr('ibeis.model.hots.smk.smk_index', 'compute_negentropy_names')"
    python -c "import utool;
    utool.print_auto_docstr('ibeis.model.hots.smk.smk_index', 'compute_negentropy_names')"
    """
    print(auto_docstr(modname, funcname))


def parse_return_type(sourcecode):
    r"""
    import utool
    sourcecode = utool.codeblock(r'''
    def foo(tmp=False):
        bar = True
        return bar
    ''')
    """

    import utool
    import ast
    if utool.VERBOSE:
        print('[utool] parsing return types')
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


def infer_arg_types_and_descriptions(argname_list, defaults):
    """
    Args:
        argname_list (list):
        defaults (?):

    Returns:
        tuple : (arg_types, argdesc_list)

    Example:
        >>> import utool
        >>> argname_list = ['ibs', 'qaid', 'fdKfds']
        >>> defaults = None
        >>> arg_types, argdesc_list = utool.infer_arg_types_and_descriptions(argname_list, defaults)
    """

    # hacks for IBEIS
    if is_developer():
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


def make_default_docstr(func):
    """
    Tries to make a sensible default docstr so the user
    can fill things in without typing too much
    """
    import utool as ut
    current_doc = inspect.getdoc(func)
    needs_surround = current_doc is None or len(current_doc) == 0
    argspec = inspect.getargspec(func)
    (argname_list, varargs, varkw, defaults) = argspec

    arg_types, argdesc_list = infer_arg_types_and_descriptions(argname_list, defaults)

    argdoc_list = [arg + ' (%s):%s' % (_type, desc)
                   for arg, _type, desc in zip(argname_list, arg_types, argdesc_list)]

    docstr_parts = []

    # Move source down to base indentation, but remember original indentation
    sourcecode = inspect.getsource(func)
    num_indent = ut.get_indentation(sourcecode)
    sourcecode = ut.unindent(sourcecode)

    # Header part
    header_block = func.func_name
    docstr_parts.append(header_block)

    def docstr_block(header, block):
        if isinstance(block, str):
            indented_block = '\n' + ut.indent(block)
        elif isinstance(block, list):
            indented_block = ut.indentjoin(block)
        else:
            assert False, 'impossible state'
        return ''.join([header, ':', indented_block])

    # Args part
    if len(argdoc_list) > 0:
        arg_header = 'Args'
        argsblock = docstr_block(arg_header, argdoc_list)
        docstr_parts.append(argsblock)

    # Return / Yeild part
    if sourcecode is not None:
        return_type, return_name, return_header = parse_return_type(sourcecode)
        if return_header is not None:
            #returndoc = (return_header + ': \n' + '    %s' % return_type)
            return_doc = return_type + ': '
            if return_name is not None:
                return_doc += return_name
            returnblock = docstr_block(return_header, return_doc)
            docstr_parts.append(returnblock)

    # Example part
    if sourcecode is not None:
        # try to generate a simple and unit testable example
        exampleheader = 'Example'
        examplecode = ut.codeblock(
            '''
            from {modname} import *  # NOQA
            '''
            #import utool as ut
        ).format(modname=func.__module__)
        # Default example values
        defaults_ = [] if defaults is None else defaults
        default_vals = ['?'] * (len(argname_list) - len(defaults_)) + list(defaults_)
        arg_val_iter = zip(argname_list, default_vals)
        argdef_lines = ['%s = %r' % (argname, val) for argname, val in arg_val_iter]
        examplecode += ut.indentjoin(argdef_lines, '\n')
        # Default example result assignment
        result_assign = ''
        result_print = None
        if 'return_name' in vars():
            if return_type is not None:
                if return_name is None:
                    return_name = 'result'
                result_assign = return_name + ' = '
                result_print = 'print(' + return_name + ')'
        # Default example call
        example_call = func.func_name + '(' + ', '.join(argname_list) + ')'
        examplecode += '\n' + result_assign + example_call
        if result_print is not None:
            examplecode += '\n' + result_print
        exampleblock = docstr_block(exampleheader, ut.indent(examplecode, '>>> '))
        docstr_parts.append(exampleblock)

    # DEBUG part
    DEBUG_DOC = False
    if DEBUG_DOC:
        debugheader = 'Debug'
        debugblock = ut.codeblock(
            '''
            num_indent = {num_indent}
            '''
        ).format(num_indent=num_indent)
        debugblock = docstr_block(debugheader, debugblock)
        docstr_parts.append(debugblock)

    # Enclosure / Indentation Parts
    if needs_surround:
        docstr_parts = ['"""'] + ['\n\n'.join(docstr_parts)] + ['"""']
        default_docstr = '\n'.join(docstr_parts)
    else:
        default_docstr = '\n\n'.join(docstr_parts)

    docstr_indent = ' ' * (num_indent + 4)
    default_docstr = ut.indent(default_docstr, docstr_indent)
    return default_docstr


def timeit_compare(stmt_list, setup='', iterations=100000, verbose=True,
                   strict=False):
    """
    Compares several statments by timing them and also
    checks that they have the same return value

    Args:
        stmt_list (list) : list of statments to compare
        setup (str) :
        iterations (int) :
        verbose (bool) :
        strict (bool) :

    Returns:
        tuple (bool, list, list) : (passed, time_list, result_list)
            passed (bool): True if all results are the same
            time_list (list): list of times for each statment
            result_list (list): list of results values for each statment

    Example:
        >>> import utool
        >>> setup = utool.unindent(
            '''
            import numpy as np
            np.random.seed(0)
            invVR_mats = np.random.rand(1000, 3, 3).astype(np.float64)
            ''')
        >>> stmt1 = 'invVR_mats[:, 0:2, 2].T'
        >>> stmt2 = 'invVR_mats.T[2, 0:2]'
        >>> iterations = 100000
        >>> verbose = True
        >>> stmt_list = [stmt1, stmt2]
        >>> utool.timeit_compare(stmt_list, setup='', iterations=1000, verbose=True)
    """
    import timeit
    import utool

    for stmtx in range(len(stmt_list)):
        # Hacky way of removing assignment and just getting statement
        # We have to make sure it is ok when using it for kwargs
        stmt = stmt_list[stmtx]
        eqpos = stmt.find('=')
        lparen_pos = stmt.find('(')
        if eqpos > 0 and (lparen_pos == -1 or lparen_pos > eqpos):
            stmt = '='.join(stmt.split('=')[1:])
            stmt_list[stmtx] = stmt

    if verbose:
        print('+----------------')
        print('| TIMEIT COMPARE')
        print('+----------------')
        print('| iterations = %d' % (iterations,))
        print('| Input:')
        #print('|     +------------')
        print('|     | num | stmt')
        for count, stmt in enumerate(stmt_list):
            print('|     | %3d | %r' % (count, stmt))
        print('...')
        sys.stdout.flush()
        #print('+     L________________')

    result_list = [testit(stmt, setup) for stmt in stmt_list]
    time_list   = [timeit.timeit(stmt, setup=setup, number=iterations)
                   for stmt in stmt_list]

    passed = utool.util_list.list_allsame(result_list)
    if verbose:
        print('| Output:')
        if not passed:
            print('|    * FAILED: results differ between some statements')
            print('| Results:')
            for result in result_list:
                for count, result in enumerate(result_list):
                    print('<Result %d>' % count)
                    print(result)
                    print('</Result %d>' % count)
            if strict:
                raise AssertionError('Results are not valid')
        else:
            print('|    * PASSED: each statement produced the same result')
            passed = True
        #print('|    +-----------------------------------')
        print('|    | num | total time | per loop | stmt')
        for count, tup in enumerate(zip(stmt_list, time_list)):
            stmt, time = tup
            print('|    | %3d | %10s | %8s | %s' %
                  (count, utool.seconds_str(time),
                   utool.seconds_str(time / iterations), stmt))
        #print('|    L___________________________________')
        if verbose:
            print('L_________________')
        return (passed, time_list, result_list)


def testit(stmt, setup):
    # Make temporary locals/globals for a sandboxlike run
    _globals = {}
    try:
        exec(setup, _globals)
    except Exception as ex:
        import utool
        print('Setup Error')
        print(setup)
        print('---')
        utool.printex(ex, 'error executing setup', keys=['setup'])
        raise
    try:
        result = eval(stmt, _globals)
    except Exception as ex:
        import utool
        print('Statement Error')
        print(setup)
        print('---')
        print(stmt)
        utool.printex(ex, 'error executing statement', keys=['stmt'])
        raise
    return result


def memory_dump():
    """
    References:
       from http://stackoverflow.com/questions/141351/how-do-i-find-what-is-using-memory-in-a-python-process-in-a-production-system
    """
    import cPickle
    dump = open("memory.pickle", 'w')
    for obj in gc.get_objects():
        i = id(obj)
        size = sys.getsizeof(obj, 0)
        #    referrers = [id(o) for o in gc.get_referrers(obj) if hasattr(o, '__class__')]
        referents = [id(o) for o in gc.get_referents(obj) if hasattr(o, '__class__')]
        if hasattr(obj, '__class__'):
            cls = str(obj.__class__)
            cPickle.dump({'id': i, 'class': cls, 'size': size, 'referents': referents}, dump)


def _disableable(func):
    def _wrp_disableable(self, *args, **kwargs):
        if self.disabled:
            return
        return func(self, *args, **kwargs)
    return _wrp_disableable


class MemoryTracker(object):
    """
    A ``class`` for tracking memory usage.
    On initialization it logs the current available (free) memory.
    Calling the report method logs the current available memory as well
    as memory usage difference w.r.t the last report.

    Example:
        >>> import utool
        >>> import numpy as np
        >>> memtrack = utool.MemoryTracker('[ENTRY]')
        >>> memtrack.report('[BEFORE_CREATE]')
        >>> arr = np.ones(128 * (2 ** 20), dtype=np.uint8)
        >>> memtrack.report('[AFTER_CREATE]')
        >>> memtrack.track_obj(arr, 'arr')
        >>> memtrack.report_objs()
        >>> memtrack.report_largest()
        >>> del arr
        >>> memtrack.report('[DELETE]')
        #>>> memtrack.report_largest()
    """
    def __init__(self, lbl='Memtrack Init', disable=True):
        self.disabled = disable  # disable by default
        self.init_nBytes = self.get_available_memory()
        self.prev_nBytes = None
        self.weakref_dict = {}  # weakref.WeakValueDictionary()
        self.weakref_dict2 = {}
        self.report(lbl)

    @_disableable
    def __call__(self, lbl=''):
        self.report(lbl=lbl)

    @_disableable
    def collect(self):
        gc.collect()

    @_disableable
    def report_largest(self):
        # Doesnt quite work yet
        import numpy as np
        import gc
        import utool
        print('reporting largest')
        obj_list = gc.get_objects()
        #simple_size_list = np.array([sys.getsizeof(obj) for obj in obj_list])
        #shortlist_size = 20
        #sortx = simple_size_list.argsort()[::-1][0:shortlist_size]
        #simple_size_sorted = simple_size_list[sortx]
        #obj_sorted = [obj_list[x] for x in sortx]
        #for obj, size in zip(obj_sorted, simple_size_sorted):
        #    print('size = %r, type(obj) = %r' % (utool.byte_str2(size), type(obj)))

        print('reporting largets ndarrays')
        ndarray_list = [obj for obj in obj_list if isinstance(obj, np.ndarray)]
        ndarray_list = [obj for obj in obj_list if str(type(obj)).find('array') > -1]
        size_list = np.array([utool.get_object_size(obj) for obj in ndarray_list])
        sortx = size_list.argsort()[::-1]
        ndarray_sorted = [ndarray_list[x] for x in sortx]
        for obj, size in zip(ndarray_sorted, size_list):
            print('size = %r, type(obj) = %r' % (utool.byte_str2(size), type(obj)))

        #size_list = [utool.get_object_size(obj) for obj in obj_list]
        pass

    @_disableable
    def report(self, lbl=''):
        from .util_str import byte_str2
        self.collect()
        nBytes = self.get_available_memory()
        print('[memtrack] +----')
        if self.prev_nBytes is not None:
            diff = self.prev_nBytes - nBytes
            print('[memtrack] | [%s] diff = %s' % (lbl, byte_str2(diff)))
        else:
            print('[memtrack] | new MemoryTracker(%s)' % (lbl,))

        total_diff = self.init_nBytes - nBytes
        print('[memtrack] | Total diff = %s' % (byte_str2(total_diff)))
        print('[memtrack] | Available Memory = %s' %  (byte_str2(nBytes),))
        self.report_objs()
        print('[memtrack] L----')
        self.prev_nBytes = nBytes

    @_disableable
    def get_available_memory(self):
        from .util_resources import available_memory
        return available_memory()

    @_disableable
    def track_obj(self, obj, name):
        oid = id(obj)
        if not isinstance(obj, weakref.ref):
            obj = weakref.ref(obj)
        #obj_weakref = weakref.ref(obj)
        self.weakref_dict[oid] = obj
        self.weakref_dict2[oid] = name
        del obj

    @_disableable
    def report_objs(self):
        if len(self.weakref_dict) == 0:
            return
        import utool
        with utool.Indenter('[memtrack] '):
            #print('[memtrack] +----')
            for oid in self.weakref_dict.iterkeys():
                obj = self.weakref_dict[oid]
                if not isinstance(obj, weakref.ref):
                    obj = weakref.ref(obj)
                name = self.weakref_dict2[oid]
                report_memsize(obj, name)
                del obj
        #print('[memtrack] L----')


def report_memsize(obj, name=None, verbose=True):
    #import types
    import utool
    if name is None:
        name = 'obj'

    if not isinstance(obj, weakref.ref):
        obj = weakref.ref(obj)

    if obj() is None:
        with utool.Indenter('|   '):
            print('+----')
            print('Memsize: ')
            print('type(%s) = %r' % (name, type(obj())))
            print('%s has been deallocated' % name)
            print('L____')
            return

    referents = gc.get_referents(obj())
    referers  = gc.get_referrers(obj())
    with utool.Indenter('|   '):
        print('+----')
        print('Memsize: ')
        print('type(%s) = %r' % (name, type(obj())))
        print('%s is using: %s' % (name, utool.get_object_size_str(obj())))
        print('%s has %d referents' % (name, len(referents)))
        print('%s has %d referers' % (name, len(referers)))
        if verbose:
            if len(referers) > 0:
                for count, referer in enumerate(referers):
                    print('  <Referer %d>' % count)
                    print('    type(referer) = %r' % type(referer))
                    try:
                        #if isinstance(referer, frames.FrameType)
                        print('    frame(referer).f_code.co_name = %s' % (referer.f_code.co_name))
                    except Exception:
                        pass
                    try:
                        #if isinstance(referer, frames.FrameType)
                        print('    func(referer).func_name = %s' % (referer.func_name))
                    except Exception:
                        pass
                    if isinstance(referer, dict):
                        print('    len(referer) = %r' % len(referer))
                        if len(referer) < 30:
                            keystr = utool.packstr(repr(referer.keys()), 60, newline_prefix='        ')
                            print('    referer.keys = %s' % (keystr),)
                    print('    id(referer) = %r' % id(referer))
                    #print('referer = ' + utool.truncate_str(repr(referer)))
                    print('  </Referer %d>' % count)
        del obj
        del referents
        del referers
        print('L____')


class InteractiveIter(object):
    """
    Choose next value interactively
    """
    def __init__(self, iterable=None, enabled=True, startx=0):
        self.enabled = enabled
        self.iterable = iterable
        self.action_keys = {
            'quit_keys': ['q', 'exit', 'quit'],
            'next_keys': ['', 'n'],
            'prev_keys': ['p'],
            'reload_keys': ['r'],
            'index_keys': ['x', 'i', 'index'],
            'ipy_keys': ['ipy', 'ipython', 'cmd'],
        }
        #self.quit_keys = ['q', 'exit', 'quit']
        #self.next_keys = ['', 'n']
        #self.prev_keys = ['p']
        #self.index_keys = ['x', 'i', 'index']
        #self.reload_keys = ['r']
        #self.ipy_keys = ['ipy', 'ipython', 'cmd']
        self.index = startx
        pass

    def __call__(self, iterable=None):
        self.iterable = iterable

    def format_msg(self, msg):
        return msg.format(**self.action_keys)

    def prompt(self):
        import utool as ut
        msg = ut.indentjoin(list(map(self.format_msg, [
            'enter {next_keys} to move to the next index',
            'enter {prev_keys} to move to the previous index',
            'enter {reload_keys} to stay at the same index',
            'enter {index_keys} to move to that index',
            'enter {ipy_keys} to start IPython',
            'enter {quit_keys} to quit',
        ])), '\n | * ')
        msg = ''.join([' +-----------', msg, '\n L-----------\n'])
        # TODO: timeout, help message
        ans = input(msg).strip()
        return ans

    def handle_ans(self, ans):
        # Quit
        if ans in self.action_keys['quit_keys']:
            raise StopIteration()
        # Prev
        elif ans in self.action_keys['prev_keys']:
            self.index -= 1
        # Next
        elif ans in self.action_keys['next_keys']:
            self.index += 1
        # Reload
        elif ans in self.action_keys['reload_keys']:
            self.index += 0
        # Index
        elif any([ans.startswith(index_key + ' ') for index_key in self.action_keys['index_keys']]):
            try:
                self.index = int(ans.split(' ')[1])
            except ValueError:
                print('Unknown ans=%r' % (ans,))
        # IPython
        elif ans in self.action_keys['ipy_keys']:
            return 'IPython'
        else:
            print('Unknown ans=%r' % (ans,))
            return False
        return True

    def __iter__(self):
        import utool as ut
        if not self.enabled:
            raise StopIteration()
        assert isinstance(self.iterable, INDEXABLE_TYPES)
        self.num_items = len(self.iterable)
        print('Begin interactive iteration over %r items' % (self.num_items))
        mark_, end_ = util_progress.log_progress(total=self.num_items, lbl='interaction: ', freq=1)
        while True:
            item = self.iterable[self.index]
            mark_(self.index)
            print('')
            yield item
            ans = self.prompt()
            action = self.handle_ans(ans)
            if action == 'IPython':
                ut.embed(N=1)
        end_()
        print('Ended interactive iteration')


def tuples_to_unique_scalars(tup_list):
    seen = {}
    def addval(tup):
        val = len(seen)
        seen[tup] = val
        return val
    scalar_list = [seen[tup] if tup in seen else addval(tup) for tup in tup_list]
    return scalar_list


def get_stats(list_, axis=None):
    """
    Args:
        list_ (listlike): values to get statistics of
        axis (int): if ``list_`` is ndarray then this specifies the axis

    Returns:
        OrderedDict: stat_dict - dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    Examples:
        >>> import numpy as np
        >>> import utool
        >>> axis = 0
        >>> list_ = np.random.rand(10, 2)
        >>> utool.get_stats(list_, axis=axis)

    SeeAlso:
        print_stats
        get_stats_str
    """
    # Assure input is in numpy format
    if isinstance(list_, np.ndarray):
        nparr = list_
    elif isinstance(list_, list):
        nparr = np.array(list_)
    else:
        list_ = list(list_)
        nparr = np.array(list_)
    # Check to make sure stats are feasible
    if len(list_) == 0:
        stat_dict = {'empty_list': True}
    else:
        # Compute stats
        min_val = nparr.min(axis=axis)
        max_val = nparr.max(axis=axis)
        mean_ = nparr.mean(axis=axis)
        std_  = nparr.std(axis=axis)
        # number of entries with min val
        nMin = np.sum(nparr == min_val, axis=axis)
        # number of entries with min val
        nMax = np.sum(nparr == max_val, axis=axis)
        stat_dict = OrderedDict(
            [('max',   np.float32(max_val)),
             ('min',   np.float32(min_val)),
             ('mean',  np.float32(mean_)),
             ('std',   np.float32(std_)),
             ('nMin',  np.int32(nMin)),
             ('nMax',  np.int32(nMax)),
             ('shape', repr(nparr.shape))])
    return stat_dict

# --- Info Strings ---


def get_stats_str(list_, newlines=False, exclude_keys=[], lbl=None):
    """
    Returns the string version of get_stats

    SeeAlso:
        print_stats
        get_stats
    """
    from utool.util_str import dict_str
    import utool as ut
    stat_dict = get_stats(list_)
    for key in exclude_keys:
        del stat_dict[key]
    stat_str  = dict_str(stat_dict, strvals=True, newlines=newlines)
    if lbl is True:
        lbl = ut.get_varname_from_stack(list_, N=1)
        stat_str = 'stats(' + lbl + ') = ' + stat_str
    #stat_strs = ['%r: %s' % (key, val) for key, val in six.iteritems(stat_dict)]
    #if newlines:
    #    indent = '    '
    #    head = '{\n' + indent
    #    sep  = ',\n' + indent
    #    tail = '\n}'
    #else:
    #    head = '{'
    #    sep = ', '
    #    tail = '}'
    #stat_str = head + sep.join(stat_strs) + tail
    return stat_str


def print_stats(list_, lbl=None, newlines=False):
    """
    Prints string representation of stat of list_

    Example:
        >>> import utool
        >>> list_ = [1, 2, 3, 4, 5]
        >>> utool.print_stats(list_)
        {max: 5.0, min: 1.0, mean: 3.0, std: 1.41421, nMin: 1, nMax: 1, shape: (5,),}

    SeeAlso:
        get_stats_str
        get_stats

    """
    if lbl is not None:
        print('Stats for %s' % lbl)
    stat_str = get_stats_str(list_, newlines=newlines)
    print(stat_str)


def npArrInfo(arr):
    """
    OLD update and refactor
    """
    from .DynamicStruct import DynStruct
    info = DynStruct()
    info.shapestr  = '[' + ' x '.join([str(x) for x in arr.shape]) + ']'
    info.dtypestr  = str(arr.dtype)
    if info.dtypestr == 'bool':
        info.bittotal = 'T=%d, F=%d' % (sum(arr), sum(1 - arr))
    elif info.dtypestr == 'object':
        info.minmaxstr = 'NA'
    elif info.dtypestr[0] == '|':
        info.minmaxstr = 'NA'
    else:
        if arr.size > 0:
            info.minmaxstr = '(%r, %r)' % (arr.min(), arr.max())
        else:
            info.minmaxstr = '(None)'
    return info


def printableType(val, name=None, parent=None):
    """
    Tries to make a nice type string for a value.
    Can also pass in a Printable parent object
    """
    if parent is not None and hasattr(parent, 'customPrintableType'):
        # Hack for non - trivial preference types
        _typestr = parent.customPrintableType(name)
        if _typestr is not None:
            return _typestr
    if isinstance(val, np.ndarray):
        info = npArrInfo(val)
        _typestr = info.dtypestr
    elif isinstance(val, object):
        _typestr = val.__class__.__name__
    else:
        _typestr = str(type(val))
        _typestr = _typestr.replace('type', '')
        _typestr = re.sub('[\'><]', '', _typestr)
        _typestr = re.sub('  *', ' ', _typestr)
        _typestr = _typestr.strip()
    return _typestr


def printableVal(val, type_bit=True, justlength=False):
    """
    Very old way of doing pretty printing. Need to update and refactor.
    """
    from . import util_dev
    # Move to util_dev
    # NUMPY ARRAY
    if type(val) is np.ndarray:
        info = npArrInfo(val)
        if info.dtypestr.startswith('bool'):
            _valstr = '{ shape:' + info.shapestr + ' bittotal: ' + info.bittotal + '}'  # + '\n  |_____'
        elif info.dtypestr.startswith('float'):
            _valstr = util_dev.get_stats_str(val)
        else:
            _valstr = '{ shape:' + info.shapestr + ' mM:' + info.minmaxstr + ' }'  # + '\n  |_____'
    # String
    elif isinstance(val, (str, unicode)):
        _valstr = '\'%s\'' % val
    # List
    elif isinstance(val, list):
        if justlength or len(val) > 30:
            _valstr = 'len=' + str(len(val))
        else:
            _valstr = '[ ' + (', \n  '.join([str(v) for v in val])) + ' ]'
    elif hasattr(val, 'get_printable') and type(val) != type:  # WTF? isinstance(val, AbstractPrintable):
        _valstr = val.get_printable(type_bit=type_bit)
    elif isinstance(val, dict):
        _valstr = '{\n'
        for val_key in val.keys():
            val_val = val[val_key]
            _valstr += '  ' + str(val_key) + ' : ' + str(val_val) + '\n'
        _valstr += '}'
    else:
        _valstr = str(val)
    if _valstr.find('\n') > 0:  # Indent if necessary
        _valstr = _valstr.replace('\n', '\n    ')
        _valstr = '\n    ' + _valstr
    _valstr = re.sub('\n *$', '', _valstr)  # Replace empty lines
    return _valstr


def myprint(input_=None, prefix='', indent='', lbl=''):
    """
    OLD PRINT FUNCTION USED WITH PRINTABLE VAL
    TODO: Refactor and update
    """
    if len(lbl) > len(prefix):
        prefix = lbl
    if len(prefix) > 0:
        prefix += ' '
    print_(indent + prefix + str(type(input_)) + ' ')
    if isinstance(input_, list):
        print(indent + '[')
        for item in iter(input_):
            myprint(item, indent=indent + '  ')
        print(indent + ']')
    elif isinstance(input_, six.string_types):
        print(input_)
    elif isinstance(input_, dict):
        print(printableVal(input_))
    else:
        print(indent + '{')
        attribute_list = dir(input_)
        for attr in attribute_list:
            if attr.find('__') == 0:
                continue
            val = str(input_.__getattribute__(attr))
            #val = input_[attr]
            # Format methods nicer
            #if val.find('built-in method'):
            #    val = '<built-in method>'
            print(indent + '  ' + attr + ' : ' + val)
        print(indent + '}')


def info(var, lbl):
    if isinstance(var, np.ndarray):
        return npinfo(var, lbl)
    if isinstance(var, list):
        return listinfo(var, lbl)


def npinfo(ndarr, lbl='ndarr'):
    info = ''
    info += (lbl + ': shape=%r ; dtype=%r' % (ndarr.shape, ndarr.dtype))
    return info


def listinfo(list_, lbl='ndarr'):
    if not isinstance(list_, list):
        raise Exception('!!')
    info = ''
    type_set = set([])
    for _ in iter(list_):
        type_set.add(str(type(_)))
    info += (lbl + ': len=%r ; types=%r' % (len(list_), type_set))
    return info


#expected_type = np.float32
#expected_dims = 5
def numpy_list_num_bits(nparr_list, expected_type, expected_dims):
    num_bits = 0
    num_items = 0
    num_elemt = 0
    bit_per_item = {
        np.float32: 32,
        np.uint8: 8
    }[expected_type]
    for nparr in iter(nparr_list):
        arr_len, arr_dims = nparr.shape
        if nparr.dtype.type is not expected_type:
            msg = 'Expected Type: ' + repr(expected_type)
            msg += 'Got Type: ' + repr(nparr.dtype)
            raise Exception(msg)
        if arr_dims != expected_dims:
            msg = 'Expected Dims: ' + repr(expected_dims)
            msg += 'Got Dims: ' + repr(arr_dims)
            raise Exception(msg)
        num_bits += len(nparr) * expected_dims * bit_per_item
        num_elemt += len(nparr) * expected_dims
        num_items += len(nparr)
    return num_bits,  num_items, num_elemt


def make_call_graph(func, *args, **kwargs):
    """ profile with pycallgraph

    Example:
        pycallgraph graphviz -- ./mypythonscript.py

    References:
        http://pycallgraph.slowchop.com/en/master/
    """
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput):
        func(*args, **kwargs)


#def runprofile(cmd, globals_=globals(), locals_=locals()):
#    """ DEPRICATE. Tries to run a function and profile it """
#    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
#    #http://www.huyng.com/posts/python-performance-analysis/
#    #Once youve gotten your code setup with the <AT>profile decorator, use kernprof.py to run your script.
#    #kernprof.py -l -v fib.py
#    import cProfile
#    print('[util] Profiling Command: ' + cmd)
#    cProfOut_fpath = 'OpenGLContext.profile'
#    cProfile.runctx( cmd, globals_, locals_, filename=cProfOut_fpath)
#    # RUN SNAKE
#    print('[util] Profiled Output: ' + cProfOut_fpath)
#    if sys.platform == 'win32':
#        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
#    else:
#        rsr_fpath = 'runsnake'
#    view_cmd = rsr_fpath + ' "' + cProfOut_fpath + '"'
#    os.system(view_cmd)
#    return True


def _memory_profile(with_gc=False):
    """
    Helper for memory debugging. Mostly just a namespace where I experiment with
    guppy and heapy.

    References:
        http://stackoverflow.com/questions/2629680/deciding-between-subprocess-multiprocessing-and-thread-in-python

    Reset Numpy Memory::
        %reset out
        %reset array
    """
    import utool
    import guppy
    if with_gc:
        garbage_collect()
    hp = guppy.hpy()
    print('[hpy] Waiting for heap output...')
    heap_output = hp.heap()
    print(heap_output)
    print('[hpy] total heap size: ' + utool.byte_str2(heap_output.size))
    utool.util_resources.memstats()
    # Graphical Browser
    #hp.pb()


def make_object_graph(obj, fpath='sample_graph.png'):
    """ memoryprofile with objgraph

    Examples:
        #import objgraph
        #objgraph.show_most_common_types()
        #objgraph.show_growth()
        #memtrack.report()
        #memtrack.report()
        #objgraph.show_growth()
        #import gc
        #gc.collect()
        #memtrack.report()
        #y = 0
        #objgraph.show_growth()
        #memtrack.report()
        #utool.embed()

    References:
        http://mg.pov.lt/objgraph/
    """
    import objgraph
    objgraph.show_most_common_types()
    #print(objgraph.by_type('ndarray'))
    #objgraph.find_backref_chain(
    #     random.choice(objgraph.by_type('ndarray')),
    #     objgraph.is_proper_module)
    objgraph.show_refs([obj], filename='ref_graph.png')
    objgraph.show_backrefs([obj], filename='backref_graph.png')


def disable_garbage_collection():
    gc.disable()


def enable_garbage_collection():
    gc.enable()


def garbage_collect():
    gc.collect()


def get_object_size(obj):
    seen = set([])
    def _get_object_size(obj):
        if (obj is None or isinstance(obj, (str, int, bool, float))):
            return sys.getsizeof(obj)

        object_id = id(obj)
        if object_id in seen:
            return 0
        seen.add(object_id)

        totalsize = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            totalsize += obj.nbytes
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            for item in obj:
                totalsize += _get_object_size(item)
        elif isinstance(obj, dict):
            try:
                for key, val in six.iteritems(obj):
                    totalsize += _get_object_size(key)
                    totalsize += _get_object_size(val)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            totalsize += _get_object_size(obj.__dict__)
            return totalsize
        return totalsize
    return _get_object_size(obj)


def print_object_size_tree(obj):
    """ Needs work """

    def _get_object_size_tree(obj, indent='', lbl='obj', seen=None):
        if (obj is None or isinstance(obj, (str, int, bool, float))):
            return [sys.getsizeof(obj)]
        object_id = id(obj)
        if object_id in seen:
            return []
        seen.add(object_id)
        size_list = [(lbl, sys.getsizeof(obj))]
        print(indent + '%s = %s ' % (lbl, str(sys.getsizeof(obj))))
        if isinstance(obj, np.ndarray):
            size_list.append(obj.nbytes)
            print(indent + '%s = %s ' % ('arr', obj.nbytes))
        elif (isinstance(obj, (tuple, list, set, frozenset))):
            for item in obj:
                size_list += _get_object_size_tree(item, indent + '   ', 'item', seen)
        elif isinstance(obj, dict):
            try:
                for key, val in six.iteritems(obj):
                    size_list += _get_object_size_tree(key, indent + '   ', key, seen)
                    size_list += _get_object_size_tree(val, indent + '   ', key, seen)
            except RuntimeError:
                print(key)
                raise
        elif isinstance(obj, object) and hasattr(obj, '__dict__'):
            size_list += _get_object_size_tree(obj.__dict__, indent + '   ', 'dict', seen)
            return size_list
        return size_list
    seen = set([])
    _get_object_size_tree(obj, '', 'obj', seen)
    del seen


def get_object_size_str(obj, lbl='', unit=None):
    from . import util_str
    nBytes = get_object_size(obj)
    if unit is None:
        sizestr = lbl + util_str.byte_str2(nBytes)
    else:
        sizestr = lbl + util_str.byte_str(nBytes, unit)
    return sizestr


def print_object_size(obj, lbl=''):
    print(get_object_size_str(obj, lbl=lbl))


def get_object_base():
    from .DynamicStruct import DynStruct
    from .util_classes import AutoReloader
    if '--min-base' in sys.argv:
        return object
    elif '--noreload-base' not in sys.argv:
        return AutoReloader
    elif '--dyn-base' in sys.argv:
        return DynStruct


def get_cython_exe():
    from . import util_cplat
    if util_cplat.WIN32:
        cython_exe = r'C:\Python27\Scripts\cython.exe'
        if not exists(cython_exe):
            cython_exe = 'cython.py'
    else:
        cython_exe = 'cython'
    return cython_exe


def compile_cython(fpath, clean=True):
    r""" Compiles a cython pyx into a shared library

    This seems broken
    compiles pyx -> pyd/dylib/so

    Examples:
        REAL SETUP.PY OUTPUT
        cythoning vtool/linalg_cython.pyx to vtool\linalg_cython.c
        C:\MinGW\bin\gcc.exe -mdll -O -Wall ^
        -IC:\Python27\Lib\site-packages\numpy\core\include ^
        -IC:\Python27\include -IC:\Python27\PC ^
        -c vtool\linalg_cython.c ^
        -o build\temp.win32-2.7\Release\vtool\linalg_cython.o

        writing build\temp.win32-2.7\Release\vtool\linalg_cython.def

        C:\MinGW\bin\gcc.exe -shared \
        -s \
        build\temp.win32-2.7\Release\vtool\linalg_cython.o \
        build\temp.win32-2.7\Release\vtool\linalg_cython.def \
        -LC:\Python27\libs \
        -LC:\Python27\PCbuild \
        -lpython27 \
        -lmsvcr90 \
        -o build\lib.win32-2.7\vtool\linalg_cython.pyd

    """
    from . import util_cplat
    from . import util_path
    import utool

    # Get autogenerated filenames
    fpath = util_path.truepath(fpath)
    dpath_, fname_ = split(fpath)
    dpath = relpath(dpath_, os.getcwd())
    fname, ext = splitext(fname_)
    # Prefer pyx over py
    if exists(fname + '.pyx'):
        fpath = fname + '.pyx'
    fname_c  = join(dpath, fname + '.c')
    fname_lib = join(dpath, fname + util_cplat.get_pylib_ext())

    print('[utool.compile_cython] fpath=%r' % (fpath,))
    print(' --- PRECHECKS --- ')
    if clean:
        utool.delete(fname_c)
        utool.delete(fname_lib)

    utool.checkpath(fpath, verbose=True, n=4)
    utool.checkpath(fname_c, verbose=True, info=False, n=4)
    utool.checkpath(fname_lib, verbose=True, info=False, n=4)

    # Cython build arguments
    cython_exe = get_cython_exe()
    if util_cplat.WIN32:
        os.environ['LDFLAGS'] = '-march=i486'
        os.environ['CFLAGS'] = '-march=i486'
        cc_exe = r'C:\MinGW\bin\gcc.exe'
        pyinclude_list = [
            r'C:\Python27\Lib\site-packages\numpy\core\include',
            r'C:\Python27\include',
            r'C:\Python27\PC',
            np.get_include()]
        pylib_list     = [
            r'C:\Python27\libs',
            r'C:\Python27\PCbuild'
            #r'C:\Python27\DLLS',
        ]
        plat_gcc_flags = ' '.join([
            '-mdll',
            '-O',
            '-DNPY_NO_DEPRECATED_API',
            #'-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
            '-Wall',
            '-Wno-unknown-pragmas',
            '-Wno-format',
            '-Wno-unused-function',
            '-m32',
            '-shared',
            #'-fPIC',
            #'-fwrapv',
        ])
        #plat_gcc_flags = ' '.join([
        #    '-shared',
        #    '-m32',
        #    '-mdll',
        #    '-march=i486',
        #    '-O',
        #])
        plat_link_flags = '-lpython27 -lmsvcr90'

#C:\MinGW\bin\gcc.exe -shared -s build\temp.win32-2.7\Release\vtool\linalg_cython.o build\temp.win32-2.7\Release\vtool\linalg_cython.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o  build\lib.win32-2.7\linalg_cython.pyd

    else:
        cc_exe = 'gcc'
        cython_exe = 'cython'
        pyinclude_list = [r'/usr/include/python2.7', np.get_include()]
        pylib_list     = []
        plat_gcc_flags = ' '.join([
            '-shared',
            '-pthread',
            '-fPIC',
            '-fwrapv',
            '-O2',
            '-Wall',
            '-fno-strict-aliasing',
        ])
    #C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\linalg_cython.c -o build\temp.win32-2.7\Release\vtool\linalg_cyth

    pyinclude = '' if len(pyinclude_list) == 0 else '-I' + ' -I'.join(pyinclude_list)
    pylib     = '' if len(pylib_list)     == 0 else '-L' + ' -L'.join(pylib_list)
    gcc_flag_list = [
        plat_gcc_flags,
        pyinclude,
        pylib,
        plat_link_flags,
    ]
    gcc_flags = ' '.join(filter(lambda x: len(x) > 0, gcc_flag_list))
    gcc_build_cmd = cc_exe + ' ' + gcc_flags + ' -o ' + fname_lib + ' -c ' + fname_c

    cython_build_cmd = cython_exe + ' ' + fpath

    # HACK
    print('\n --- CYTHON_COMMANDS ---')
    print(utool.pack_into(cython_build_cmd, textwidth=80, newline_prefix='  '))
    print('')
    print(utool.pack_into(gcc_build_cmd, textwidth=80, newline_prefix='  '))
    print(gcc_build_cmd)
    print('\n --- COMMAND_EXECUTION ---')

    def verbose_cmd(cmd):
        print('\n<CMD>')
        print(cmd)
        ret = os.system(cmd)
        print('> ret = %r' % ret)
        print('</CMD>\n')
        #print('-------------------')
        return ret

    ret = verbose_cmd(cython_build_cmd)
    assert utool.checkpath(fname_c, verbose=True, n=2), 'failed cython build'
    ret = verbose_cmd(gcc_build_cmd)
    assert utool.checkpath(fname_lib, verbose=True, n=2), 'failed gcc cython build'
    #try:
    #    #lib_dpath, lib_fname = split(fname_lib)
    #    #cwd = os.getcwd()
    #    #os.chdir(lib_dpath)
    #    ##exec('import ' + splitext(lib_fname)[0])
    #    #os.chdir(cwd)
    #    pass
    #except Exception:
    #    pass
    #    raise

    #out, err, ret = util_cplat.shell(cython_exe + ' ' + fpath)
    #out, err, ret = util_cplat.shell((cython_exe, fpath))
    #if ret == 0:
    #    out, err, ret = util_cplat.shell(cc_exe + ' ' + gcc_flags + ' -o ' + fname_so + ' ' + fname_c)
    return ret


def find_exe(name, path_hints=[], required=True):
    from . import util_cplat
    if util_cplat.WIN32 and not name.endswith('.exe'):
        name += '.exe'

    for path in path_hints:
        exe_fpath = join(path, name)
        if exists(exe_fpath):
            return exe_fpath

    if required:
        raise AssertionError('cannot find ' + name)


def _on_ctrl_c(signal, frame):
    print('Caught ctrl+c')
    sys.exit(0)


def init_catch_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)


def reset_catch_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior


def is_developer():
    import utool
    return utool.get_computer_name().lower() in ['hyrule', 'ooo', 'bakerstreet']
