# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ast
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
from utool import util_class
from utool._internal import meta_util_six
print, rrr, profile = util_inject.inject2(__name__)


VERBOSE_INSPECT, VERYVERB_INSPECT = util_arg.get_module_verbosity_flags('inspect')


LIB_PATH = dirname(os.__file__)


#def check_dynamic_member_vars(self):
#    return {name: attr for name, attr in self.__dict__.items()
#            if not name.startswith("__") and not callable(attr) and not type(attr) is staticmethod}


@util_class.reloadable_class
class BaronWraper(object):
    def __init__(self, sourcecode):
        import redbaron
        import utool as ut
        with ut.Timer('building baron'):
            self.baron = redbaron.RedBaron(sourcecode)

    def print_diff(self, fpath=None):
        import utool as ut
        fpath = getattr(self, 'fpath', fpath)
        assert fpath is not None, 'specify original file'
        old_text = ut.readfrom(fpath)
        new_text = self.to_string()
        diff_text = ut.difftext(old_text, new_text, 1)
        colored_diff_text = ut.color_diff_text(diff_text)
        print(colored_diff_text)

    def write(self, fpath=None):
        import utool as ut
        fpath = getattr(self, 'fpath', fpath)
        assert fpath is not None, 'specify original file'
        new_text = self.to_string()
        ut.write_to(fpath, new_text)

    @classmethod
    def from_fpath(cls, fpath):
        import utool as ut
        sourcecode = ut.readfrom(fpath)
        self = cls(sourcecode)
        self.fpath = fpath
        return self

    def to_string(self):
        text = self.baron.dumps()
        return text

    def defined_functions(self, recursive=True):
        found = self.baron.find_all('def', recursive=recursive)
        return found
        #name_list = [node.name for node in found]
        #return name_list

    def find_usage(self, name):
        found = self.baron.find_all('NameNode', value=name, recursive=True)
        used_in = []
        for node in found:
            parent_func = self.find_root_function(node)
            used_in.append(parent_func)
        return used_in

    def find_func(self, name):
        return self.baron.find('def', name=name)

    def find_root_function(self, node):
        par = node.parent_find('def')
        if par is None:
            raise ValueError('no parent for node=%r' % (node.name))
        elif par.indentation == '':
            # Top level function
            return par
        elif par.parent is not None:
            par2 = par.parent
            if par2.type == 'class' and par2.indentation == '':
                return par
            else:
                return self.find_root_function(par)
        else:
            raise ValueError('unknown error for node=%r' % (node.name))

    def internal_call_graph(self, with_doctests=False):
        """
        """
        import utool as ut
        import networkx as nx
        G = nx.DiGraph()
        with ut.Timer('finding defed funcs'):
            functions = self.defined_functions()

        with ut.Timer('parsing docstrings'):
            doc_nodes = {}
            for func in functions:
                if with_doctests and len(func) > 0:
                    if func[0].type == 'raw_string':
                        docstr_ = eval(func[0].value)
                        docstr = ut.unindent(docstr_)
                        docblocks = ut.parse_docblocks_from_docstr(docstr)
                        count = 0
                        for key, block in docblocks:
                            if key.startswith('Example'):
                                doctest = block
                                # docblocks['Example:']
                                docname = '<doctest%d>' % (count,) + func.name
                                doc_nodes[docname] = doctest
                                G.add_node(docname)
                                G.nodes[docname]['color'] = (1, 0, 0)
                                count += 1

        with ut.Timer('building function call graph'):
            func_names = [func.name for func in functions]
            for callee in func_names:
                #G.nodes[callee]['color'] = '0x000000'
                G.add_node(six.text_type(callee))
            import re
            pat = re.compile(ut.regex_or(func_names))
            found = self.baron.find_all('NameNode', value=pat, recursive=True)

        for node in ut.ProgIter(found, 'Searching for parent funcs', adjust=False, freq=1):
            parent_func = self.find_root_function(node)
            caller = parent_func.name
            callee = node.name
            G.add_edge(six.text_type(callee), six.text_type(caller))

        with ut.Timer('building doctest call graph'):
            for func in ut.ProgIter(functions, lbl='doctest call graph'):
                # Check for usage in doctests
                for caller, doctest in doc_nodes.items():
                    if func.name in doctest:
                        #G.add_edge(caller, callee)
                        G.add_edge(callee, caller)
            return G


def get_internal_call_graph(fpath, with_doctests=False):
    """
    CommandLine:
        python -m utool.util_inspect get_internal_call_graph --show --modpath=~/code/ibeis/ibeis/init/main_helpers.py --show
        python -m utool.util_inspect get_internal_call_graph --show --modpath=~/code/dtool/dtool/depcache_table.py --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> fpath = ut.get_argval('--modpath', default='.')
        >>> with_doctests = ut.get_argflag('--with_doctests')
        >>> G = get_internal_call_graph(fpath, with_doctests)
        >>> ut.quit_if_noshow()
        >>> import plottool_ibeis as pt
        >>> pt.qt4ensure()
        >>> pt.show_nx(G, fontsize=8, as_directed=False)
        >>> z = pt.zoom_factory()
        >>> p = pt.pan_factory()
        >>> ut.show_if_requested()
    """
    import utool as ut
    fpath = ut.truepath(fpath)
    sourcecode = ut.readfrom(fpath)
    self = ut.BaronWraper(sourcecode)
    G = self.internal_call_graph(with_doctests=with_doctests)
    return G


def get_module_from_class(class_):
    return sys.modules[class_.__module__]


def check_static_member_vars(class_, fpath=None, only_init=True):
    """
    class_ can either be live object or a classname

    # fpath = ut.truepath('~/code/ibeis/ibeis/viz/viz_graph2.py')
    # classname = 'AnnotGraphWidget'
    """
    #import ast
    #import astor
    import utool as ut

    if isinstance(class_, six.string_types):
        classname = class_
        if fpath is None:
            raise Exception('must specify fpath')
    else:
        # We were given a live object
        if not isinstance(class_, type):
            # We were given the class instance not the class
            class_instance = class_
            class_ = class_instance.__class__
        classname = class_.__name__
        if fpath is None:
            module = ut.get_module_from_class(class_)
            fpath = ut.get_modpath(module)

    sourcecode = ut.readfrom(fpath)

    import redbaron
    # Pares a FULL syntax tree that keeps blockcomments
    baron = redbaron.RedBaron(sourcecode)

    for node in baron:
        if node.type == 'class' and node.name == classname:
            classnode = node
            break

    def find_parent_method(node):
        par = node.parent_find('def')
        if par is not None and par.parent is not None:
            if par.parent.type == 'class':
                return par
            else:
                return find_parent_method(par)

    # TODO: Find inherited attrs
    #classnode.inherit_from
    # inhertied_attrs = ['parent']
    # inhertied_attrs = []

    class_methods = []
    for node in classnode:
        if node.type == 'def':
            if only_init:
                if node.name == '__init__':
                    class_methods.append(node)
            else:
                class_methods.append(node)

    class_vars = []
    self_vars = []
    for method_node in class_methods:
        self_var = method_node.arguments[0].dumps()
        self_vars.append(self_var)
        for assign in method_node.find_all('assignment'):
            # method_node = find_parent_method(assign)
            if assign.target.dumps().startswith(self_var + '.'):
                class_vars.append(assign.target.value[1].dumps())
    static_attrs = ut.unique(class_vars)
    return static_attrs

    # class_members = ut.unique(class_vars + class_methods + inhertied_attrs)
    if False:
        self_var = self_vars[0]

        # Find everything that is used
        complex_cases = []
        simple_cases = []
        all_self_ref = classnode.find_all(
            'name_', value=re.compile('.*' + self_var + '\\.*'))
        for x in all_self_ref:
            if x.parent.type == 'def_argument':
                continue
            if x.parent.type == 'atomtrailers':
                atom = x.parent
                if ut.depth(atom.fst()) <= 3:
                    simple_cases.append(atom)
                else:
                    complex_cases.append(atom)
                #print(ut.depth(atom.value.data))
                #print(atom.value)
                #print(atom.dumps())
                #if len(atom.dumps()) > 200:
                #    break

        accessed_attrs = []
        for x in simple_cases:
            if x.value[0].dumps() == self_var:
                attr = x.value[1].dumps()
                accessed_attrs.append(attr)
        accessed_attrs = ut.unique(accessed_attrs)

        ut.setdiff(accessed_attrs, class_vars)

        # print('Missing Attrs: ' + str(ut.setdiff(accessed_attrs, class_members)))
        #fst = baron.fst()
        #node = (baron.node_list[54])  # NOQA
        #[n.type for n in baron.node_list]
        #generator = astor.codegen.SourceGenerator(' ' * 4)
        #generator.visit(pt)
        #resturctured_source = (''.join(generator.result))
        #print(resturctured_source)
        #visitor = ast.NodeVisitor()
        #visitor.visit(pt)
        #class SpecialVisitor(ast.NodeVisitor):


def get_funcnames_from_modpath(modpath, include_methods=True):
    """
    Get all functions defined in module
    """
    import utool as ut
    if True:
        import jedi
        source = ut.read_from(modpath)
        #script = jedi.Script(source=source, source_path=modpath, line=source.count('\n') + 1)
        definition_list = jedi.names(source)
        funcname_list = [definition.name for definition in definition_list if definition.type == 'function']
        if include_methods:
            classdef_list = [definition for definition in definition_list if definition.type == 'class']
            defined_methods = ut.flatten([definition.defined_names() for definition in classdef_list])
            funcname_list += [method.name for method in defined_methods
                              if method.type == 'function' and not method.name.startswith('_')]
    else:
        import redbaron
        # Pares a FULL syntax tree that keeps blockcomments
        sourcecode = ut.read_from(modpath)
        baron = redbaron.RedBaron(sourcecode)
        funcname_list = [node.name for node in baron.find_all('def', recursive=include_methods)
                         if not node.name.startswith('_')]
    return funcname_list


#@profile
def check_module_usage(modpath_patterns):
    """
    FIXME: not fully implmented

    Desired behavior is ---
    Given a set of modules specified by a list of patterns, returns how the
    functions defined in the modules are called: a) with themselves and b) by
    other files in the project not in the given set.

    Args:
        modpath_patterns (list):

    CommandLine:
        python -m utool.util_inspect check_module_usage --show
        utprof.py -m utool.util_inspect check_module_usage --show
        python -m utool.util_inspect check_module_usage --pat="['auto*', 'user_dialogs.py', 'special_query.py', 'qt_inc_automatch.py', 'devcases.py']"
        python -m utool.util_inspect check_module_usage --pat="preproc_detectimg.py"
        python -m utool.util_inspect check_module_usage --pat="neighbor_index.py"
        python -m utool.util_inspect check_module_usage --pat="manual_chip_funcs.py"
        python -m utool.util_inspect check_module_usage --pat="preproc_probchip.py"
        python -m utool.util_inspect check_module_usage --pat="guiback.py"

        python -m utool.util_inspect check_module_usage --pat="util_str.py"

    Ignore:

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> modpath_patterns = ['_grave*']
        >>> modpath_patterns = ['auto*', 'user_dialogs.py', 'special_query.py', 'qt_inc_automatch.py', 'devcases.py']
        >>> modpath_patterns = ['neighbor_index.py']
        >>> modpath_patterns = ['manual_chip_funcs.py']
        >>> modpath_patterns = ut.get_argval('--pat', type_=list, default=['*'])
        >>> result = check_module_usage(modpath_patterns)
        >>> print(result)
    """
    import utool as ut
    #dpath = '~/code/ibeis/ibeis/algo/hots'
    modpaths = ut.flatten([ut.glob_projects(pat) for pat in modpath_patterns])
    modpaths = ut.unique(modpaths)
    modnames = ut.lmap(ut.get_modname_from_modpath, modpaths)
    print('Checking usage of modules: ' + ut.repr3(modpaths))

    # Mark as True is module is always explicitly imported
    restrict_to_importing_modpaths = False
    cache = {}

    def find_where_module_is_imported(modname):
        """ finds where a module was explicitly imported. (in most scenareos) """
        # Find places where the module was imported
        patterns = ut.possible_import_patterns(modname)
        # do modname grep with all possible import patterns
        grepres = ut.grep_projects(patterns, new=True, verbose=False, cache=cache)
        return grepres.found_fpath_list

    def find_function_callers(funcname, importing_modpaths):
        """ searches for places where a function is used """
        pattern = '\\b' + funcname + '\\b',
        # Search which module uses each public member
        grepres = ut.grep_projects(
            pattern, new=True, verbose=False, cache=cache,
            fpath_list=importing_modpaths)
        # Exclude places where function is defined or call is commented out
        nohit_patterns = [
            r'^\s*def',
            r'^\s*#',
            r'\-\-exec\-',
            r'\-\-test-',
            r'^\s*python -m ',
            r'^\s*python -m ibeis ',
            r'^\s*ibeis ',
            r'\-\-test\-[a-zA-z]*\.',
            r'\-\-exec\-[a-zA-z]*\.',
        ]
        nohit_patterns += [
            r'^\s*\>\>\>',
        ]
        filter_pat = ut.regex_or(nohit_patterns)
        # import copy
        # grepres_ = copy.deepcopy(grepres)
        grepres.inplace_filter_results(filter_pat)
        grepres.found_modnames = ut.lmap(ut.get_modname_from_modpath,
                                         grepres.found_fpath_list)
        parent_numlines = ut.lmap(len, grepres.found_lines_list)

        numcall_graph_ = dict(zip(grepres.found_modnames, parent_numlines))
        # Remove self references
        #ut.delete_keys(numcall_graph_, modnames)
        return numcall_graph_, grepres

    print('Find modules that use this the query modules')
    # Note: only works for explicit imports
    importing_modpaths_list = [find_where_module_is_imported(modname) for modname in modnames]
    print('Find members of the query modules')
    funcnames_list = [get_funcnames_from_modpath(modpath) for modpath in modpaths]

    print('Building call graph')
    cache = {}
    func_numcall_graph = ut.ddict(dict)
    grep_results = ut.ddict(dict)
    # Extract public members from each module
    exclude_self = ut.get_argflag('--exclude-self')
    _iter = list(zip(modnames, modpaths, importing_modpaths_list, funcnames_list))
    _iter = ut.ProgIter(_iter, lbl='Searching query module', bs=False)
    for modname, modpath, importing_modpaths, funcname_list in _iter:
        if not restrict_to_importing_modpaths:
            importing_modpaths = None

        # Search for each function in modpath
        for funcname in ut.ProgIter(funcname_list, lbl='Searching funcs in query module'):
            numcall_graph_, grepres = find_function_callers(funcname, importing_modpaths)
            grep_results[modname][funcname] = grepres
            if exclude_self:
                if modname in numcall_graph_:
                    del numcall_graph_[modname]
            func_numcall_graph[modname][funcname] = numcall_graph_

    # Sort by incidence cardinality
    # func_numcall_graph = ut.odict([(key, ut.sort_dict(val, 'vals', len)) for key, val in func_numcall_graph.items()])
    # Sort by weighted degree
    func_numcall_graph = ut.odict([(key, ut.sort_dict(val, 'vals', lambda x: sum(x.values())))
                                   for key, val in func_numcall_graph.items()])
    # Print out grep results in order
    print('PRINTING GREP RESULTS IN ORDER')
    for modname, num_callgraph in func_numcall_graph.items():
        print('\n============\n')
        for funcname in num_callgraph.keys():
            print('\n============\n')
            with ut.Indenter('[%s]' % (funcname,)):
                grepres = grep_results[modname][funcname]
                print(grepres)
                # print(func_numcall_graph[modname][funcname])
    print('PRINTING NUMCALLGRAPH IN ORDER')
    # Print out callgraph in order
    print('func_numcall_graph = %s' % (ut.repr3(func_numcall_graph),))

    # importance_dict = {}
    # import copy
    # func_call_graph2 = copy.deepcopy(func_numcall_graph)
    # #ignore_modnames = []
    # ignore_modnames = ['ibeis.algo.hots.multi_index', 'ibeis.algo.hots._neighbor_experiment']
    # num_callers = ut.ddict(dict)
    # for modname, modpath in list(zip(modnames, modpaths)):
    #     subdict = func_call_graph2[modname]
    #     for funcname in subdict.keys():
    #         numcall_graph_ = subdict[funcname]
    #         ut.delete_keys(numcall_graph_, modnames)
    #         ut.delete_keys(numcall_graph_, ignore_modnames)
    #         num_callers[modname][funcname] = sum(numcall_graph_.values())
    #     print(ut.repr4(num_callers[modname], sorted_=True, key_order_metric='val'))

    # # Check external usage
    # unused_external = []
    # grep_results2 = copy.deepcopy(grep_results)
    # for modname, grepres_subdict in grep_results2.items():
    #     for funcname, grepres_ in grepres_subdict.items():
    #         idxs = ut.find_list_indexes(grepres_.found_modnames, modnames)
    #         idxs += ut.find_list_indexes(grepres_.found_modnames, ignore_modnames)
    #         idxs = list(ut.filter_Nones(idxs))
    #         ut.delete_items_by_index(grepres_, idxs)
    #         ut.delete_items_by_index(grepres_.found_modnames, idxs)
    #         if len(grepres_) > 0:
    #             print(grepres_.make_resultstr())
    #         else:
    #             unused_external += [funcname]

    # print('internal grep')
    # # Check internal usage
    # unused_internal = []
    # grep_results2 = copy.deepcopy(grep_results)
    # for modname, grepres_subdict in grep_results2.items():
    #     for funcname, grepres_ in grepres_subdict.items():
    #         idxs = ut.filter_Nones(ut.find_list_indexes(grepres_.found_modnames, [modname]))
    #         idxs_ = ut.index_complement(idxs, len(grepres_.found_modnames))
    #         ut.delete_items_by_index(grepres_, idxs_)
    #         ut.delete_items_by_index(grepres_.found_modnames, idxs_)
    #         grepres_.hack_remove_pystuff()
    #         #self = grepres_
    #         if len(grepres_) > 0:
    #             #print(modname)
    #             #print(funcname)
    #             #print(grepres_.extended_regex_list)
    #             print(grepres_.make_resultstr())
    #         else:
    #             unused_internal += [funcname]

    # # HACK: how to write ut.parfor
    # # returns a 0 lenth iterator so the for loop is never run. Then uses code
    # # introspection to determine the content of the for loop body executes code
    # # using the values of the local variables in a parallel / distributed
    # # context.

    # for modname, modpath in zip(modnames, modpaths):
    #     pattern = '\\b' + modname + '\\b',
    #     grepres = ut.grep_projects(pattern, new=True, verbose=False, cache=cache)
    #     parent_modnames = ut.lmap(ut.get_modname_from_modpath, grepres.found_fpath_list)
    #     parent_numlines = ut.lmap(len, grepres.found_lines_list)
    #     importance = dict(zip(parent_modnames, parent_numlines))
    #     ut.delete_keys(importance, modnames)
    #     importance_dict[modname] = importance

    # print('importance_dict = %s' % (ut.repr3(importance_dict),))
    # combo = reduce(ut.dict_union, importance_dict.values())
    # print('combined %s' % (ut.repr3(combo),))
    # print(ut.repr3(found_fpath_list))
    pass


def get_object_methods(obj):
    """
    Returns all methods belonging to an object instance specified in by the
    __dir__ function

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> obj = ut.NiceRepr()
        >>> methods1 = ut.get_object_methods()
        >>> ut.inject_func_as_method(obj, ut.get_object_methods)
        >>> methods2 = ut.get_object_methods()
        >>> assert ut.get_object_methods in methods2
    """
    import utool as ut
    attr_list = (getattr(obj, attrname) for attrname in dir(obj))
    methods = [attr for attr in attr_list if ut.is_method(attr)]
    return methods


def help_members(obj, use_other=False):
    r"""
    Inspects members of a class

    Args:
        obj (class or module):

    CommandLine:
        python -m utool.util_inspect help_members

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> obj = ut.DynStruct
        >>> result = help_members(obj)
        >>> print(result)
    """
    import utool as ut
    attrnames = dir(obj)
    attr_list = [getattr(obj, attrname) for attrname in attrnames]
    attr_types = ut.lmap(ut.type_str, map(type, attr_list))
    unique_types, groupxs = ut.group_indices(attr_types)
    type_to_items = ut.dzip(unique_types, ut.apply_grouping(attr_list, groupxs))
    type_to_itemname = ut.dzip(unique_types, ut.apply_grouping(attrnames, groupxs))
    #if memtypes is None:
    #    memtypes = list(type_to_items.keys())
    memtypes = ['instancemethod']  # , 'method-wrapper']
    func_mems = ut.dict_subset(type_to_items, memtypes, [])

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

    if use_other:
        other_mems = ut.delete_keys(type_to_items.copy(), memtypes)
        other_mems_attrnames = ut.dict_subset(type_to_itemname, other_mems.keys())
        named_other_attrs = ut.dict_union_combine(other_mems_attrnames, other_mems, lambda x, y: list(zip(x, y)))
        print(ut.repr4(named_other_attrs, nl=2, strvals=True))


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
        # (r'img\d*', ('ndarray[uint8_t, ndim=2]', 'image data')),
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
        # (r'dict_?\d?' , ('dict', None)),
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
        belongs = False
        if hasattr(val, '__module__'):
            belongs = val.__module__ == module.__name__
        elif hasattr(val, 'func_globals'):
            belongs = val.func_globals['__name__'] == module.__name__
        if belongs:
            list_.append(val)
    return list_


def zzz_profiled_is_no():
    pass


@profile
def zzz_profiled_is_yes():
    pass


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
        python -m utool --tf iter_module_doctestable \
            --modname=ibeis.algo.hots.chip_match
            --modname=ibeis.control.IBEISControl
            --modname=ibeis.control.SQLDatabaseControl
            --modname=ibeis.control.manual_annot_funcs
            --modname=ibeis.control.manual_annot_funcs
            --modname=ibeis.expt.test_result
            --modname=utool.util_progress --debug-key=build_msg_fmtstr_time2
            --modname=utool.util_progress --debug-key=ProgressIter

    Debug:
       # fix profile with doctest
       utprof.py -m utool --tf iter_module_doctestable --modname=utool.util_inspect --debugkey=zzz_profiled_is_yes
       utprof.py -m utool --tf iter_module_doctestable --modname=ibeis.algo.hots.chip_match --debugkey=to_json

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *   # NOQA
        >>> import utool as ut
        >>> modname = ut.get_argval('--modname', type_=str, default=None)
        >>> kwargs = ut.argparse_funckw(iter_module_doctestable)
        >>> module = ut.util_tests if modname is None else ut.import_modname(modname)
        >>> debug_key = ut.get_argval('--debugkey', type_=str, default=None)
        >>> kwargs['debug_key'] = debug_key
        >>> kwargs['include_inherited'] = True
        >>> doctestable_list = list(iter_module_doctestable(module, **kwargs))
        >>> func_names = sorted(ut.take_column(doctestable_list, 0))
        >>> print(ut.repr4(func_names))
    """
    import ctypes

    types.BuiltinFunctionType
    valid_func_types = [
        types.FunctionType, types.BuiltinFunctionType, classmethod,
        staticmethod,
        types.MethodType,  # handles classmethod
        # types.BuiltinMethodType,
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
            print('DEBUG')
            print('debug_key = %r' % (debug_key,))
            exec('item = val')
            # import utool as ut
            # ut.embed()
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
                        if not include_inherited and not is_defined_by_module(subval, module, parent=val):
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


def is_defined_by_module2(item, module):
    belongs = False
    if hasattr(item, '__module__'):
        belongs = item.__module__ == module.__name__
    elif hasattr(item, 'func_globals'):
        belongs = item.func_globals['__name__'] == module.__name__
    return belongs


def is_defined_by_module(item, module, parent=None):
    """
    Check if item is directly defined by a module.
    This check may be prone to errors.
    """
    flag = False
    if isinstance(item, types.ModuleType):
        if not hasattr(item, '__file__'):
            try:
                # hack for cv2 and xfeatures2d
                import utool as ut
                name = ut.get_modname_from_modpath(module.__file__)
                flag = name in str(item)
            except Exception:
                flag = False
        else:
            item_modpath = os.path.realpath(dirname(item.__file__))
            mod_fpath = module.__file__.replace('.pyc', '.py')
            if not mod_fpath.endswith('__init__.py'):
                flag = False
            else:
                modpath = os.path.realpath(dirname(mod_fpath))
                modpath = modpath.replace('.pyc', '.py')
                flag = item_modpath.startswith(modpath)
    elif hasattr(item, '_utinfo'):
        # Capture case where there is a utool wrapper
        orig_func = item._utinfo['orig_func']
        flag = is_defined_by_module(orig_func, module, parent)
    else:
        if isinstance(item, staticmethod):
            # static methods are a wrapper around a function
            item = item.__func__
        try:
            func_globals = meta_util_six.get_funcglobals(item)
            func_module_name = func_globals['__name__']
            if func_module_name == 'line_profiler':
                valid_names = dir(module)
                if parent is not None:
                    valid_names += dir(parent)
                if item.func_name in valid_names:
                    # hack to prevent small names
                    #if len(item.func_name) > 8:
                    if len(item.func_name) > 6:
                        flag = True
            elif func_module_name == module.__name__:
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
        python -m utool._internal.meta_util_six is_builtin

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
        >>> # DISABLE_DOCTEST
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
        >>> # DISABLE_DOCTEST
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
    foo2 = kwargs['foo2']
    foobar = str(foo) + str(bar) + str(foo2)
    return foobar


def get_kwdefaults2(func, parse_source=False):
    return get_kwdefaults(func, parse_source=True)


def six_get_argspect(func):
    """
    Old getargspec-like interface
    """
    if six.PY2:
        argspec = inspect.getargspec(func)
    else:
        if hasattr(inspect, 'getfullargspec'):
            from collections import namedtuple
            ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
            fullargspec = inspect.getfullargspec(func)
            argspec = ArgSpec(fullargspec.args, fullargspec.varargs,
                              fullargspec.varkw, fullargspec.defaults)
        else:
            argspec = inspect.getargspec(func)
    return argspec


def get_kwdefaults(func, parse_source=False):
    r"""
    Args:
        func (func):

    Returns:
        dict:

    CommandLine:
        python -m utool.util_inspect get_kwdefaults

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> func = dummy_func
        >>> parse_source = True
        >>> kwdefaults = get_kwdefaults(func, parse_source)
        >>> print('kwdefaults = %s' % (ut.repr4(kwdefaults),))
    """
    #import utool as ut
    #with ut.embed_on_exception_context:
    argspec = six_get_argspect(func)
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
    argspec = six_get_argspect(func)
    argnames = argspec.args
    return argnames


def get_funcname(func):
    return meta_util_six.get_funcname(func)


def set_funcname(func, newname):
    return meta_util_six.set_funcname(func, newname)


def get_method_func(func):
    if six.PY2:
        return func.im_func
    else:
        return func.__func__


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


def get_func_docblocks(func_or_class):
    import utool as ut
    docstr = ut.get_docstr(func_or_class)
    docblocks = ut.parse_docblocks_from_docstr(docstr)
    return docblocks


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


def special_parse_process_python_code(sourcecode):
    r"""
    pip install redbaron
    http://stackoverflow.com/questions/7456933/python-ast-with-preserved-comments

    CommandLine:
        python -m utool.util_inspect special_parse_process_python_code --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> sourcecode = ut.read_from(ut.util_inspect.__file__)
        >>> result = special_parse_process_python_code(sourcecode)
        >>> print(result)
    """
    import ast
    import astor
    #sourcecode = 'from __future__ import print_function\n' + sourcecode
    sourcecode_ = sourcecode.encode('utf8')
    pt = ast.parse(sourcecode_, 'testfile')

    generator = astor.codegen.SourceGenerator(' ' * 4)
    generator.visit(pt)
    resturctured_source = (''.join(generator.result))
    print(resturctured_source)

    visitor = ast.NodeVisitor()
    visitor.visit(pt)

    import redbaron
    # Pares a FULL syntax tree that keeps blockcomments
    baron = redbaron.RedBaron(sourcecode)
    #fst = baron.fst()
    node = (baron.node_list[54])  # NOQA
    [n.type for n in baron.node_list]

    #class SpecialVisitor(ast.NodeVisitor):


def parse_function_names(sourcecode, top_level=True, ignore_condition=1):
    """
    Finds all function names in a file without importing it

    Args:
        sourcecode (str):

    Returns:
        list: func_names

    CommandLine:
        python -m utool.util_inspect parse_function_names

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> fpath = ut.util_inspect.__file__.replace('.pyc', '.py')
        >>> #fpath = ut.truepath('~/code/bintrees/bintrees/avltree.py')
        >>> sourcecode = ut.readfrom(fpath)
        >>> func_names = parse_function_names(sourcecode)
        >>> result = ('func_names = %s' % (ut.repr2(func_names),))
        >>> print(result)
    """
    import ast
    import utool as ut
    func_names = []
    if six.PY2:
        sourcecode = ut.ensure_unicode(sourcecode)
        encoded = sourcecode.encode('utf8')
        pt = ast.parse(encoded)
    else:
        pt = ast.parse(sourcecode)

    class FuncVisitor(ast.NodeVisitor):

        def __init__(self):
            super(FuncVisitor, self).__init__()
            self.condition_names = None
            self.condition_id = -9001
            self.in_condition_chain = False

        def visit_If(self, node):
            if ignore_condition:
                return
            # if ignore_conditional:
            #     return
            # Ignore the main statement
            # print('----')
            # print('node.test = {!r}'.format(node.test))
            # print('node.orelse = {!r}'.format(node.orelse))
            if _node_is_main_if(node):
                return

            # if isinstance(node.orelse, ast.If):
            #     # THIS IS AN ELIF
            #     self.condition_id += 1
            #     self.in_condition_chain = True
            #     ast.NodeVisitor.generic_visit(self, node)
            #     self.in_condition_chain = False
            #     pass
            # # TODO: where does else get parsed exactly?

            # Reset the set of conditionals
            # self.condition_id = 0
            # self.condition_names = ut.ddict(list)

            # self.in_condition_chain = True
            ast.NodeVisitor.generic_visit(self, node)
            # self.in_condition_chain = False

            # if False:
            #     # IF THIS WAS AN ELSE:
            #     if self.condition_names is not None:
            #         # anything defined in all conditions is kosher
            #         from six.moves import reduce
            #         common_names = reduce(set.intersection,
            #                               map(set, self.condition_names.values()))
            #         self.func_names.extend(common_names)
            #         self.condition_names = None

        def visit_FunctionDef(self, node):
            # if self.in_condition_chain and self.condition_names is not None:
            #     # dont immediately add things in conditions. Wait until we can
            #     # ensure which definitions are common in all conditions.
            #     self.condition_names[self.condition_id].append(node.name)
            # else:
            func_names.append(node.name)
            if not top_level:
                ast.NodeVisitor.generic_visit(self, node)

        def visit_ClassDef(self, node):
            if not top_level:
                ast.NodeVisitor.generic_visit(self, node)
    try:
        FuncVisitor().visit(pt)
    except Exception:
        raise
        pass
    return func_names


def _node_is_main_if(node):
    if isinstance(node.test, ast.Compare):
        try:
            if all([
                isinstance(node.test.ops[0], ast.Eq),
                node.test.left.id == '__name__',
                node.test.comparators[0].s == '__main__',
            ]):
                return True
        except Exception:
            pass
    return False


def parse_project_imports(dpath):
    """
    dpath = ub.truepath('~/code/clab/clab')

    Script:
        >>> dpath = ut.get_argval('--dpath')
        >>> parse_project_imports()

    """
    import ubelt as ub
    import glob
    from os.path import join, exists
    package_modules = set()
    for fpath in glob.glob(join(dpath, '**/*.py'), recursive=True):
        try:
            sourcecode = ub.readfrom(fpath)
            _, modules = ut.parse_import_names(sourcecode, False, fpath=fpath)
            for mod in modules:
                package_modules.add(mod.split('.')[0])  # just bases
            if 'clab/live' in package_modules:
                raise ValueError()
                break
        except SyntaxError:
            print('encountered SyntaxError in fpath = {!r}'.format(fpath))
    import warnings
    import inspect
    stdlibs = [dirname(warnings.__file__), dirname(inspect.__file__)]
    def is_module_batteries_included(m):
        if m in sys.builtin_module_names:
            return True
        for p in stdlibs:
            if exists(join(p, m + '.py')):
                return True
            if exists(join(p, m)):
                return True
    used_modules = sorted([m for m in package_modules if not is_module_batteries_included(m)])
    print('used_modules non-buildin modules = {}'.format(ub.repr2(used_modules)))


def parse_import_names(sourcecode, top_level=True, fpath=None, branch=False):
    """
    Finds all function names in a file without importing it

    Args:
        sourcecode (str):

    Returns:
        list: func_names

    CommandLine:
        python -m utool.util_inspect parse_import_names

    References:
        https://stackoverflow.com/questions/20445733/how-to-tell-which-modules-have-been-imported-in-some-source-code

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> fpath = ut.util_inspect.__file__.replace('.pyc', '.py')
        >>> #fpath = ut.truepath('~/code/bintrees/bintrees/avltree.py')
        >>> sourcecode = ut.readfrom(fpath)
        >>> func_names = parse_import_names(sourcecode)
        >>> result = ('func_names = %s' % (ut.repr2(func_names),))
        >>> print(result)
    """
    import ast
    import_names = []
    if six.PY2:
        import utool as ut
        sourcecode = ut.ensure_unicode(sourcecode)
        encoded = sourcecode.encode('utf8')
        pt = ast.parse(encoded)
    else:
        pt = ast.parse(sourcecode)

    modules = []

    class ImportVisitor(ast.NodeVisitor):

        def _parse_alias_list(self, aliases):
            for alias in aliases:
                if alias.asname is not None:
                    import_names.append(alias.asname)
                else:
                    if '.' not in alias.name:
                        import_names.append(alias.name)

        def visit_Import(self, node):
            self._parse_alias_list(node.names)
            self.generic_visit(node)

            for alias in node.names:
                modules.append(alias.name)

        def visit_ImportFrom(self, node):
            self._parse_alias_list(node.names)
            self.generic_visit(node)

            for alias in node.names:
                prefix = ''
                if node.level:
                    if fpath is not None:
                        from xdoctest import static_analysis as static
                        modparts = static.split_modpath(os.path.abspath(fpath))[1].replace('\\', '/').split('/')
                        parts = modparts[:-node.level]
                        # parts = os.path.split(static.split_modpath(os.path.abspath(fpath))[1])[:-node.level]
                        prefix = '.'.join(parts) + '.'
                        # prefix = '.'.join(os.path.split(fpath)[-node.level:]) + '.'
                    else:
                        prefix = '.' * node.level
                # modules.append(node.level * '.' + node.module + '.' + alias.name)
                # modules.append(prefix + node.module + '.' + alias.name)
                modules.append(prefix + node.module)

        def visit_FunctionDef(self, node):
            # Ignore modules imported in functions
            if not top_level:
                self.generic_visit(node)
                # ast.NodeVisitor.generic_visit(self, node)

        def visit_ClassDef(self, node):
            if not top_level:
                self.generic_visit(node)
                # ast.NodeVisitor.generic_visit(self, node)

        def visit_If(self, node):
            if not branch:
                # TODO: determine how to figure out if a name is in all branches
                if not _node_is_main_if(node):
                    # Ignore the main statement
                    self.generic_visit(node)
    try:
        ImportVisitor().visit(pt)
    except Exception:
        pass
    return import_names, modules


def find_funcs_called_with_kwargs(sourcecode, target_kwargs_name='kwargs'):
    r"""
    Finds functions that are called with the keyword `kwargs` variable

    CommandLine:
        python3 -m utool.util_inspect find_funcs_called_with_kwargs

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> sourcecode = ut.codeblock(
                '''
                x, y = list(zip(*ut.ichunks(data, 2)))
                somecall(arg1, arg2, arg3=4, **kwargs)
                import sys
                sys.badcall(**kwargs)
                def foo():
                    bar(**kwargs)
                    ut.holymoly(**kwargs)
                    baz()
                    def biz(**kwargs):
                        foo2(**kwargs)
                ''')
        >>> child_funcnamess = ut.find_funcs_called_with_kwargs(sourcecode)
        >>> print('child_funcnamess = %r' % (child_funcnamess,))
        >>> assert 'foo2' not in child_funcnamess, 'foo2 should not be found'
        >>> assert 'bar' in child_funcnamess, 'bar should be found'
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
        ie, know when kwargs is passed to these functions and
        then look assume the object that was updated is a dictionary
        and check wherever that is passed to kwargs as well.
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
                #print(ut.repr4(node.__dict__,))
            if isinstance(node.func, ast.Attribute):
                try:
                    funcname = node.func.value.id + '.' + node.func.attr
                except AttributeError:
                    funcname = None
            elif isinstance(node.func, ast.Name):
                funcname = node.func.id
            else:
                raise NotImplementedError(
                    'do not know how to parse: node.func = %r' % (node.func,))
            if six.PY2:
                kwargs = node.kwargs
                kwargs_name = None if kwargs is None else kwargs.id
                if funcname is not None and kwargs_name == target_kwargs_name:
                    child_funcnamess.append(funcname)
                if debug:
                    print('funcname = %r' % (funcname,))
                    print('kwargs_name = %r' % (kwargs_name,))
            else:
                if node.keywords:
                    for kwargs in node.keywords:
                        if kwargs.arg is None:
                            if hasattr(kwargs.value, 'id'):
                                kwargs_name = kwargs.value.id
                                if funcname is not None and kwargs_name == target_kwargs_name:
                                    child_funcnamess.append(funcname)
                                if debug:
                                    print('funcname = %r' % (funcname,))
                                    print('kwargs_name = %r' % (kwargs_name,))
            ast.NodeVisitor.generic_visit(self, node)
    try:
        KwargParseVisitor().visit(pt)
    except Exception:
        raise
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
        python -m utool.util_inspect parse_return_type
        python -m utool.util_inspect --test-parse_return_type


    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> sourcecode = ut.codeblock(
        ...     'def foo(tmp=False):\n'
        ...     '    bar = True\n'
        ...     '    return bar\n'
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
        ...     'def foo(tmp=False):\n'
        ...     '    return True\n'
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
        ...     'def foo(tmp=False):\n'
        ...     '    for i in range(2): \n'
        ...     '        yield i\n'
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
        ...     'def foo(tmp=False):\n'
        ...     '    if tmp is True:\n'
        ...     '        return (True, False)\n'
        ...     '    elif tmp is False:\n'
        ...     '        return 1\n'
        ...     '    else:\n'
        ...     '        bar = baz()\n'
        ...     '        return bar\n'
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


def exec_func_src(func, globals_=None, locals_=None, key_list=None,
                  sentinal=None, update=None, keys=None, verbose=False,
                  start=None, stop=None):
    """
    execs a func and returns requested local vars.

    Does not modify globals unless update=True (or in IPython)

    SeeAlso:
        ut.execstr_funckw
    """
    if keys is None:
        keys = key_list
    import utool as ut
    sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True)
    if update is None:
        update = ut.inIPython()
    if globals_ is None:
        globals_ = ut.get_parent_frame().f_globals
    if locals_ is None:
        locals_ = ut.get_parent_frame().f_locals
    if sentinal is not None:
        sourcecode = ut.replace_between_tags(sourcecode, '', sentinal)
    if start is not None or stop is not None:
        sourcecode = '\n'.join(sourcecode.splitlines()[slice(start, stop)])
    globals_new = globals_.copy()
    if locals_ is not None:
        globals_new.update(locals_)
        # globals_new.update({k: v for k, v in locals_.items()
        #                     if k not in globals_new})
    orig_globals = globals_new.copy()
    #six.exec_(sourcecode, globals_new, locals_)
    if verbose:
        print(ut.color_text(sourcecode, 'python'))
    six.exec_(sourcecode, globals_new)
    # Draw intermediate steps
    if keys is None:
        #return locals_
        # Remove keys created in function execution
        ut.delete_keys(globals_new, orig_globals.keys())
        if update:
            # update input globals?
            globals_.update(globals_new)
        # ~~ TODO autodetermine the keys from the function vars
        return globals_new
    else:
        if update:
            # update input globals?
            globals_.update(globals_new)
        #var_list = ut.dict_take(locals_, keys)
        var_list = ut.dict_take(globals_new, keys)
        return var_list


def exec_func_src2(func, globals_=None, locals_=None, sentinal=None,
                   verbose=False, start=None, stop=None):
    """
    execs a func and returns requested local vars.

    Does not modify globals unless update=True (or in IPython)

    SeeAlso:
        ut.execstr_funckw
    """
    import utool as ut
    sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True)
    if globals_ is None:
        globals_ = ut.get_parent_frame().f_globals
    if locals_ is None:
        locals_ = ut.get_parent_frame().f_locals
    if sentinal is not None:
        sourcecode = ut.replace_between_tags(sourcecode, '', sentinal)
    if start is not None or stop is not None:
        sourcecode = '\n'.join(sourcecode.splitlines()[slice(start, stop)])
    if verbose:
        print(ut.color_text(sourcecode, 'python'))
    # TODO: find the name of every variable that was assigned in the function
    # and get it from the context
    locals2_ = locals_.copy()
    globals2_ = globals_.copy()
    six.exec_(sourcecode, globals2_, locals2_)
    return locals2_


def exec_func_src3(func, globals_, sentinal=None, verbose=False,
                   start=None, stop=None):
    """
    execs a func and returns requested local vars.

    Does not modify globals unless update=True (or in IPython)

    SeeAlso:
        ut.execstr_funckw
    """
    import utool as ut
    sourcecode = ut.get_func_sourcecode(func, stripdef=True, stripret=True)
    if sentinal is not None:
        sourcecode = ut.replace_between_tags(sourcecode, '', sentinal)
    if start is not None or stop is not None:
        sourcecode = '\n'.join(sourcecode.splitlines()[slice(start, stop)])
    if verbose:
        print(ut.color_text(sourcecode, 'python'))
    six.exec_(sourcecode, globals_)


def execstr_func_doctest(func, num=0, start_sentinal=None, end_sentinal=None):
    """
    execs a func doctest and returns requested local vars.
        >>> from utool.util_inspect import *  # NOQA

    func = encoder.learn_threshold2
    num = 0
    start_sentinal = 'import plottool_ibeis as pt'
    end_sentinal = 'pnum_ = pt.make_pnum_nextgen'
    """
    import utool as ut
    docsrc = ut.get_doctest_examples(func)[num][0]
    lines = docsrc.split('\n')
    if start_sentinal is None:
        linex1 = 0
    else:
        linex1 = ut.where([x.startswith(start_sentinal) for x in lines])[0]
    if end_sentinal is None:
        linex2 = len(lines)
    else:
        linex2 = ut.where([x.startswith(end_sentinal) for x in lines])[0]
    docsrc_part = '\n'.join(lines[linex1:linex2])
    return docsrc_part


def exec_func_doctest(func, start_sentinal=None, end_sentinal=None, num=0, globals_=None, locals_=None):
    """
    execs a func doctest and returns requested local vars.

    func = encoder.learn_threshold2
    num = 0
    start_sentinal = 'import plottool_ibeis as pt'
    end_sentinal = 'pnum_ = pt.make_pnum_nextgen'
    """
    import utool as ut
    docsrc_part = execstr_func_doctest(func, num, start_sentinal, end_sentinal)
    if globals_ is None:
        globals_ = ut.get_parent_frame().f_globals
    if locals_ is None:
        locals_ = ut.get_parent_frame().f_locals
    globals_new = globals_.copy()
    if locals_ is not None:
        globals_new.update(locals_)
    print("EXEC PART")
    print(ut.highlight_code(docsrc_part))
    six.exec_(docsrc_part, globals_new)


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
    elif sourcefile is not None and (sourcefile != '<string>'):
        try_limit = 2
        for num_tries in range(try_limit):
            try:
                #print(func)
                sourcecode = inspect.getsource(func)
                if not isinstance(sourcecode, six.text_type):
                    sourcecode = sourcecode.decode('utf-8')
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
        patern = '(' + regex_decor + ')?' + regex_defline
        nodef_source = ut.regex_replace(patern, '', sourcecode)
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
    argspec = six_get_argspect(func)
    # try:
    # except Exception:
    #     argspec = inspect.getfullargspec(func)
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
            print(six_get_argspect(func))

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
    argspec = six_get_argspect(func)
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

    Ignore:
        >>> attrname = funcname
        >>> namespace = mod.__dict__


        >>> import utool as ut
        >>> globals_ = ut.util_inspect.__dict__
        >>> attrname = 'KWReg.print_defaultkw'
    """
    #subdict = meta_util_six.get_funcglobals(root_func)
    subtup = attrname.split('.')
    subdict = namespace
    for attr in subtup[:-1]:
        subdict = subdict[attr].__dict__
    leaf_name = subtup[-1]
    leaf_attr = subdict[leaf_name]
    return leaf_attr


def recursive_parse_kwargs(root_func, path_=None, verbose=None):
    """
    recursive kwargs parser
    TODO: rectify with others
    FIXME: if docstr indentation is off, this fails

    SeeAlso:
        argparse_funckw
        recursive_parse_kwargs
        parse_kwarg_keys
        parse_func_kwarg_keys
        get_func_kwargs

    Args:
        root_func (function):  live python function
        path_ (None): (default = None)

    Returns:
        list:

    CommandLine:
        python -m utool.util_inspect recursive_parse_kwargs:0
        python -m utool.util_inspect recursive_parse_kwargs:0 --verbinspect
        python -m utool.util_inspect recursive_parse_kwargs:1

        python -m utool.util_inspect recursive_parse_kwargs:2 --mod vtool --func ScoreNormalizer.visualize

        python -m utool.util_inspect recursive_parse_kwargs:2 --mod ibeis.viz.viz_matches --func show_name_matches --verbinspect
        python -m utool.util_inspect recursive_parse_kwargs:2 --mod ibeis.expt.experiment_drawing --func draw_rank_cmc --verbinspect

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
        >>> # xdoctest: +REQUIRES(module:ibeis)
        >>> from utool.util_inspect import *  # NOQA
        >>> from ibeis.algo.hots import chip_match
        >>> import utool as ut
        >>> recursive_parse_kwargs(chip_match.ChipMatch.show_ranked_matches)
        >>> recursive_parse_kwargs(chip_match.ChipMatch)

        import ibeis
        import utool as ut
        ibs = ibeis.opendb(defaultdb='testdb1')
        kwkeys1 = ibs.parse_annot_stats_filter_kws()
        ut.recursive_parse_kwargs(ibs.get_annotconfig_stats, verbose=1)
        kwkeys2 = list(ut.recursive_parse_kwargs(ibs.get_annotconfig_stats).keys())

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> modname = ut.get_argval('--mod', type_=str, default='plottool_ibeis')
        >>> funcname = ut.get_argval('--func', type_=str, default='draw_histogram')
        >>> mod = ut.import_modname(modname)
        >>> root_func = lookup_attribute_chain(funcname, mod.__dict__)
        >>> path_ = None
        >>> parsed = recursive_parse_kwargs(root_func)
        >>> flags = ut.unique_flags(ut.take_column(parsed, 0))
        >>> unique = ut.compress(parsed, flags)
        >>> print('parsed = %s' % (ut.repr4(parsed),))
        >>> print('unique = %s' % (ut.repr4(unique),))
    """
    if verbose is None:
        verbose = VERBOSE_INSPECT
    if verbose:
        print('[inspect] recursive parse kwargs root_func = %r ' % (root_func,))

    import utool as ut
    if path_ is None:
        path_ = []
    if root_func in path_:
        if verbose:
            print('[inspect] Encountered cycle. returning')
        return []
    path_.append(root_func)
    spec = ut.get_func_argspec(root_func)
    # ADD MORE
    kwargs_list = []
    found_explicit = list(ut.get_kwdefaults(root_func, parse_source=False).items())
    if verbose:
        print('[inspect] * Found explicit %r' % (found_explicit,))

    #kwargs_list = [(kw,) for kw in  ut.get_kwargs(root_func)[0]]
    sourcecode = ut.get_func_sourcecode(root_func, strip_docstr=True,
                                        stripdef=True)
    sourcecode1 = ut.get_func_sourcecode(root_func, strip_docstr=True,
                                         stripdef=False)
    found_implicit = ut.parse_kwarg_keys(sourcecode1, spec.keywords,
                                         with_vals=True)
    if verbose:
        print('[inspect] * Found found_implicit %r' % (found_implicit,))
    kwargs_list = found_explicit + found_implicit

    def hack_lookup_mod_attrs(attr):
        # HACKS TODO: have find_funcs_called_with_kwargs infer an attribute is a
        # module / function / type. In the module case, we can import it and
        # look it up.  Maybe args, or returns can help infer type.  Maybe just
        # register some known varnames.  Maybe jedi has some better way to do
        # this.
        if attr == 'ut':
            subdict = ut.__dict__
        elif attr == 'pt':
            import plottool_ibeis as pt
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
            if ut.is_func_or_method(subdict):
                # Was subdict supposed to be named something else here?
                subfunc = subdict
                return subfunc
        except (KeyError, TypeError):
            for attr in subtup[:-1]:
                try:
                    subdict = subdict[attr].__dict__
                except (KeyError, TypeError):
                    # limited support for class lookup
                    if ut.is_method(root_func) and spec.args[0] == attr:
                        if six.PY2:
                            subdict = root_func.im_class.__dict__
                        else:
                            subdict = root_func.__class__.__dict__
                    else:
                        # FIXME TODO lookup_attribute_chain
                        subdict = hack_lookup_mod_attrs(attr)
                        if subdict is None:
                            print('Unable to find attribute of attr=%r' % (attr,))
                            if ut.SUPER_STRICT:
                                raise
        if subdict is not None:
            attr_name = subtup[-1]
            subfunc = subdict[attr_name]
        else:
            subfunc = None
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
                print('Unable to find function definition subfunc_name=%r' %
                      (subfunc_name,))
                if ut.SUPER_STRICT:
                    raise
                subfunc = None
        if subfunc is not None:
            subkw_list = recursive_parse_kwargs(subfunc, path_, verbose=verbose)
            new_subkw = subkw_list
            # have_keys = set(ut.take_column(kwargs_list, 0))
            # new_subkw = [item for item in subkw_list
            #              if item[0] not in have_keys]
        else:
            new_subkw = []
        return new_subkw

    if spec.keywords is not None:
        if verbose:
            print('[inspect] Checking spec.keywords=%r' % (spec.keywords,))
        subfunc_name_list = ut.find_funcs_called_with_kwargs(sourcecode, spec.keywords)
        if verbose:
            print('[inspect] Checking subfunc_name_list with len {}'.format(len(subfunc_name_list)))
        for subfunc_name in subfunc_name_list:
            try:
                new_subkw = check_subfunc_name(subfunc_name)
                if verbose:
                    print('[inspect] * Found %r' % (new_subkw,))
                kwargs_list.extend(new_subkw)
            except (TypeError, Exception):
                print('warning: unable to recursivley parse type of : %r' % (subfunc_name,))
    return kwargs_list


def get_funckw(func, recursive=True):
    import utool as ut
    funckw_ = ut.get_func_kwargs(func, recursive=recursive)
    # if recursive:
    #     funckw_.update(dict(ut.recursive_parse_kwargs(func)))
    return funckw_


def parse_func_kwarg_keys(func, with_vals=False):
    """ hacky inference of kwargs keys

    SeeAlso:
        argparse_funckw
        recursive_parse_kwargs
        parse_kwarg_keys
        parse_func_kwarg_keys
        get_func_kwargs

    """
    sourcecode = get_func_sourcecode(func, strip_docstr=True,
                                     strip_comments=True)
    kwkeys = parse_kwarg_keys(sourcecode, with_vals=with_vals)
    #ut.get_func_kwargs  TODO
    return kwkeys


def get_func_kwargs(func, recursive=True):
    """
    func = ibeis.run_experiment

    SeeAlso:
        argparse_funckw
        recursive_parse_kwargs
        parse_kwarg_keys
        parse_func_kwarg_keys
        get_func_kwargs
    """
    import utool as ut
    argspec = ut.get_func_argspec(func)
    if argspec.defaults is None:
        header_kw = {}
    else:
        header_kw = dict(zip(argspec.args[::-1], argspec.defaults[::-1]))
    if argspec.keywords is not None:
        header_kw.update(dict(ut.recursive_parse_kwargs(func)))
    return header_kw


def parse_kwarg_keys(source, keywords='kwargs', with_vals=False, debug='auto'):
    r"""
    Parses the source code to find keys used by the `**kwargs` keywords
    dictionary variable. if `with_vals` is True, we also attempt to infer the
    default values.

    Args:
        source (str):

    Returns:
        list: kwarg_keys

    CommandLine:
        python -m utool.util_inspect parse_kwarg_keys

        python -m utool.util_inspect parse_kwarg_keys

    SeeAlso:
        argparse_funckw
        recursive_parse_kwargs
        parse_kwarg_keys
        parse_func_kwarg_keys
        get_func_kwargs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> source = (
        >>>           "\n x = 'hidden_x'"
        >>>           "\n y = 3 # hidden val"
        >>>           "\n kwargs.get(x, y)"
        >>>           "\n kwargs.get('foo', None)\n kwargs.pop('bar', 3)"
        >>>           "\n kwargs.pop('str', '3fd')\n kwargs.pop('str', '3f\\'d')"
        >>>           "\n \"kwargs.get('baz', None)\"\n kwargs['foo2']"
        >>>           "\n #kwargs.get('biz', None)\""
        >>>           "\n kwargs['bloop']"
        >>>           "\n x = 'bop' in kwargs"
        >>>           )
        >>> print('source = %s\n' % (source,))
        ...
        >>> ut.exec_funckw(parse_kwarg_keys, globals())
        >>> with_vals = True
        >>> kwarg_items = parse_kwarg_keys(source, with_vals=with_vals, debug=0)
        >>> result = ('kwarg_items = %s' % (ut.repr2(kwarg_items, nl=1),))
        >>> print(result)
        kwarg_items = [
            ('foo', None),
            ('bar', 3),
            ('str', '3fd'),
            ('str', "3f'd"),
            ('foo2', None),
            ('bloop', None),
        ]
        >>> kwarg_keys = ut.take_column(kwarg_items, 0)
        >>> assert 'baz' not in kwarg_keys
        >>> assert 'foo' in kwarg_keys
        >>> assert 'bloop' in kwarg_keys
        >>> assert 'bop' not in kwarg_keys
    """
    import utool as ut
    import ast
    sourcecode = 'from __future__ import print_function, unicode_literals\n' + ut.unindent(source)
    pt = ast.parse(sourcecode)
    kwargs_items = []
    if debug == 'auto':
        debug = VERYVERB_INSPECT
    target_kwargs_name = keywords

    if debug:
        import astor
        print('\nInput:')
        print('target_kwargs_name = %r' % (target_kwargs_name,))
        print('\nSource:')
        print(sourcecode)
        print('\nParse:')
        print(astor.dump(pt))

    class KwargParseVisitor(ast.NodeVisitor):
        """
        TODO: understand ut.update_existing and dict update
        ie, know when kwargs is passed to these functions and
        then look assume the object that was updated is a dictionary
        and check wherever that is passed to kwargs as well.

        Other visit_<classname> values:
            http://greentreesnakes.readthedocs.io/en/latest/nodes.html
        """
        def __init__(self):
            super(KwargParseVisitor, self).__init__()
            self.const_lookup = {}
            self.first = True

        def visit_FunctionDef(self, node):
            if debug:
                print('VISIT FunctionDef node = %r' % (node,))
                # print('node.args.kwarg = %r' % (node.args.kwarg,))
            if six.PY2:
                kwarg_name = node.args.kwarg
            else:
                if node.args.kwarg is None:
                    kwarg_name = None
                else:
                    kwarg_name = node.args.kwarg.arg

            # Record any constants defined in function definitions
            defaults_vals = node.args.defaults
            offset = len(node.args.args) - len(defaults_vals)
            default_keys = node.args.args[offset:]
            for kwname, kwval in zip(default_keys, defaults_vals):
                # try:
                if six.PY2:
                    if isinstance(kwval, ast.Name):
                        val = eval(kwval.id, {}, {})
                        self.const_lookup[kwname.id] = val
                else:
                    if isinstance(kwval, ast.NameConstant):
                        val = kwval.value
                        self.const_lookup[kwname.arg] = val
                # except Exception:
                #     pass

            if self.first or kwarg_name != target_kwargs_name:
                # target kwargs is still in scope
                ast.NodeVisitor.generic_visit(self, node)
                # always visit the first function
                self.first = False

        def visit_Subscript(self, node):
            if debug:
                print('VISIT SUBSCRIPT node = %r' % (node,))
                # print(ut.repr4(node.__dict__,))
            if isinstance(node.value, ast.Name):
                if node.value.id == target_kwargs_name:
                    if six.PY3 and isinstance(node.slice, ast.Constant):
                        index = node.slice
                        key = index.value
                        item = (key, None)
                        kwargs_items.append(item)
                    elif isinstance(node.slice, ast.Index):
                        index = node.slice
                        key = index.value
                        if isinstance(key, ast.Str):
                            # item = (key.s, None)
                            item = (key.s, None)
                            kwargs_items.append(item)
                        elif six.PY3 and isinstance(key, ast.Constant):
                            # item = (key.s, None)
                            item = (key.value, None)
                            kwargs_items.append(item)

        @staticmethod
        def _eval_bool_op(val):
            # Can we handle this more intelligently?
            val_value = None
            if isinstance(val.op, ast.Or):
                if any([isinstance(x, ast.NameConstant) and x.value is True for x in val.values]):
                    val_value = True
            elif isinstance(val.op, ast.And):
                if any([isinstance(x, ast.NameConstant) and x.value is False for x in val.values]):
                    val_value = False
            return val_value

        def visit_Call(self, node):
            if debug:
                print('VISIT Call node = %r' % (node,))
                # print(ut.repr4(node.__dict__,))
            if isinstance(node.func, ast.Attribute):
                try:
                    objname = node.func.value.id
                except AttributeError:
                    return
                methodname = node.func.attr
                # funcname = objname + '.' + methodname
                if objname == target_kwargs_name and methodname in {'get', 'pop'}:
                    args = node.args
                    if len(args) == 2:
                        key, val = args
                        if isinstance(key, ast.Name):
                            # TODO lookup constant
                            pass
                        elif isinstance(key, ast.Str):
                            key_value = key.s
                            val_value = None   # ut.NoParam
                            if isinstance(val, ast.Str):
                                val_value = val.s
                            elif isinstance(val, ast.Num):
                                val_value = val.n
                            elif isinstance(val, ast.Name):
                                if val.id == 'None':
                                    val_value = None
                                else:
                                    val_value = self.const_lookup.get(
                                            val.id, None)
                                    # val_value = 'TODO lookup const'
                                    # TODO: lookup constants?
                                    pass
                            elif six.PY3:
                                if isinstance(val, ast.NameConstant):
                                    val_value = val.value
                                elif isinstance(val, ast.Call):
                                    val_value = None
                                elif isinstance(val, ast.BoolOp):
                                    val_value = self._eval_bool_op(val)
                                elif isinstance(val, ast.Dict):
                                    if len(val.keys) == 0:
                                        val_value = {}
                                    else:
                                        val_value = {}
                                    # val_value = callable
                                else:
                                    print('Warning: util_inspect doent know how to parse {}'.format(repr(val)))
                            item = (key_value, val_value)
                            kwargs_items.append(item)
            ast.NodeVisitor.generic_visit(self, node)
    try:
        KwargParseVisitor().visit(pt)
    except Exception:
        raise
        pass
    if with_vals:
        return kwargs_items
    else:
        return ut.take_column(kwargs_items, 0)


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
        print(ut.repr4(kwreg.defaultkw))


def get_instance_attrnames(obj, default=True, **kwargs):
    cls = obj.__class__
    out = []
    for a in dir(cls):
        unbound_attr = getattr(cls, a, None)
        if kwargs.get('with_properties', default) and isinstance(unbound_attr, property):
            out.append(a)
        if kwargs.get('with_methods', default) and isinstance(unbound_attr, types.MethodType):
            out.append(a)
    return out


def argparse_funckw(func, defaults={}, **kwargs):
    """
    allows kwargs to be specified on the commandline from testfuncs

    Args:
        func (function):

    Kwargs:
        lbl, verbose, only_specified, force_keys, type_hint, alias_dict

    Returns:
        dict: funckw

    CommandLine:
        python -m utool.util_inspect argparse_funckw

    SeeAlso:
        exec_funckw
        recursive_parse_kwargs
        parse_kwarg_keys

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> func = get_instance_attrnames
        >>> funckw = argparse_funckw(func)
        >>> result = ('funckw = %s' % (ut.repr3(funckw),))
        >>> print(result)
        funckw = {
            'default': True,
            'with_methods': True,
            'with_properties': True,
        }
    """
    import utool as ut
    funckw_ = ut.get_funckw(func, recursive=True)
    funckw_.update(defaults)
    funckw = ut.argparse_dict(funckw_, **kwargs)
    return funckw


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
        >>> # ENABLE_DOCTEST
        >>> from utool.util_inspect import *  # NOQA
        >>> import utool as ut
        >>> func = ut.infer_function_info
        >>> #func = ut.Timer.tic
        >>> func = ut.make_default_docstr
        >>> funcinfo = infer_function_info(func)
        >>> result = ut.repr4(funcinfo.__dict__)
        >>> print(result)

    Ignore:
        >>> # DISABLE_DOCTEST
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
        >>> result = ut.repr4(funcinfo.__dict__)
        >>> print(result)
    """
    import utool as ut
    import re

    # TODO: allow a jedi argument
    if False:
        from jedi.evaluate import docstrings
        script = func.script
        argname_list = [p.name.value for p in func.params]
        argtype_list = [docstrings.follow_param(script._evaluator, p) for p in func.params]

    if isinstance(func, property):
        func = func.fget
    try:
        doc_shortdesc = ''
        doc_longdesc = ''

        known_arginfo = ut.ddict(dict)

        current_doc = inspect.getdoc(func)
        docstr_blocks = ut.parse_docblocks_from_docstr(current_doc)
        docblock_types = ut.take_column(docstr_blocks, 0)
        docblock_types = [re.sub('Example[0-9]', 'Example', type_)
                          for type_ in docblock_types]
        docblock_dict = ut.group_items(docstr_blocks, docblock_types)

        if '' in docblock_dict:
            docheaders = docblock_dict['']
            docheaders_lines = ut.take_column(docheaders, 1)
            docheaders_order = ut.take_column(docheaders, 2)
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
                    except Exception:
                        print('---')
                        print('argline = \n%s' % (argline,))
                        print('---')
                        raise Exception('Unable to parse argline=%s' % (argline,))
                    #print('groupdict_ = %s' % (ut.repr4(groupdict_),))
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
            flags = ut.unique_flags(ut.take_column(kwarg_items, 0))
            kwarg_items = ut.compress(kwarg_items, flags)
            kwarg_keys = ut.take_column(kwarg_items, 0)
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
        ], tb=True)
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
    # if new
    funcinfo.va_name = varargs
    funcinfo.kw_name = varkw
    funcinfo.kw_keys = kwarg_keys
    # else
    funcinfo.varargs = varargs
    funcinfo.varkw = varkw
    # fi
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


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_inspect --enableall
        python -m utool.util_inspect --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
