#!/usr/bin/env python2
"""
#python -c "import cyth, doctest; print(doctest.testmod(cyth.cyth_script))"
python -m doctest cyth_script.py
ut
python cyth/cyth_script.py ~/code/fpath
cyth_script.py ~/code/ibeis/ibeis/model/hots
cyth_script.py "~/code/vtool/vtool"

"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, map
import utool
import sys
from os.path import isfile
from cyth import cyth_helpers
import ast
#import codegen  # NOQA
import astor
import re
import doctest
from copy import deepcopy
import cyth  # NOQA
import timeit

#class CythTransformer(ast.NodeTransformer):
#    #


#BASE_CLASS = codegen.SourceGenerator
#BASE_CLASS = ast.NodeVisitor
#BASE_CLASS = astor.ExplicitNodeVisitor
BASE_CLASS = astor.codegen.SourceGenerator


# See astor/codegen.py for details
# https://github.com/berkerpeksag/astor/blob/master/astor/codegen.py


def is_docstring(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)


"""
<CYTH>
import ast
import astor
</CYTH>
"""


def replace_funcalls(source, funcname, replacement):
    """
    >>> from cyth_script import *
    >>> replace_funcalls('foo(5)', 'foo', 'bar')
    'bar(5)'
    >>> replace_funcalls('foo(5)', 'bar', 'baz')
    'foo(5)'

    <CYTH>
    </CYTH>
    """

    # FIXME: !!!
    # http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html#nested-class-declarations
    # C++ allows nested class declaration. Class declarations can also be nested in Cython:
    # Note that the nested class is declared with a cppclass but without a cdef.
    class FunctioncallReplacer(ast.NodeTransformer):
        def visit_Call(self, node):
            """CYTH></CYTH>"""
            if isinstance(node.func, ast.Name) and node.func.id == funcname:
                node.func.id = replacement
            return node
    generator = astor.codegen.SourceGenerator(' ' * 4)
    generator.visit(FunctioncallReplacer().visit(ast.parse(source)))
    return ''.join(generator.result)
    #return ast.dump(tree)


def get_doctest_examples(source):
    # parse list of docstrings
    comment_iter = doctest.DocTestParser().parse(source)
    # remove any non-doctests
    example_list = [c for c in comment_iter if isinstance(c, doctest.Example)]
    return example_list
    #example_list = filter(lambda c: isinstance(c, doctest.Example), comment_iter)
    #return filter(lambda x: isinstance(x, doctest.Example), doctest.DocTestParser().parse(source))

def is_test(src, funcname):
    pt = ast.parse(src)
    assert isinstance(pt, ast.Module), type(pt)
#    print('<<<')
#    try:
#        print(len(pt.body) == 1)
#        print(isinstance(pt.body[0], ast.Expr))
#        print(isinstance(pt.body[0].value, ast.Call))
#        print(isinstance(pt.body[0].value.func, ast.Name))
#        print(pt.body[0].value.func.id)
#    except:
#        pass
#    print('>>>')
    try:
        # look for the only line of src to contain just a function call, and check the name
        line = pt.body[0]
        return line.value.func.id == funcname
    except AttributeError, IndexError:
        return False

def make_benchmarks(funcname, docstring):
    r"""
    >>> from cyth.cyth_script import *
    >>> funcname = 'replace_funcalls'
    >>> docstring =  '''
    ...         >>> from cyth_script import *
    ...         >>> replace_funcalls('foo(5)', 'foo', 'bar')
    ...         'bar(5)'
    ...         >>> replace_funcalls('foo(5)', 'bar', 'baz')
    ...         'foo(5)'
    ... '''
    >>> benchmark_list = list(make_benchmarks(funcname, docstring))
    >>> output = [((x.source, x.want), y.source, y.want) for x, y in benchmark_list]
    >>> print(output)
    [(('from cyth_script import *\n', ''), 'from cyth_script import *', ''), (("replace_funcalls('foo(5)', 'foo', 'bar')\n", "'bar(5)'\n"), "_replace_funcalls_cyth('foo(5)', 'foo', 'bar')", "'bar(5)'\n"), (("replace_funcalls('foo(5)', 'bar', 'baz')\n", "'foo(5)'\n"), "_replace_funcalls_cyth('foo(5)', 'bar', 'baz')", "'foo(5)'\n")]
    """
    doctest_examples = get_doctest_examples(docstring)
    #[print("doctest_examples[%d] = (%r, %r)" % (i, x, y)) for (i, (x, y)) in
    #    enumerate(map(lambda x: (x.source, x.want), doctest_examples))]
#    tweaked_examples = []
#    # would this be clearer with map?
#    for example in doctest_examples:
#        tweaked_example = deepcopy(example)
#        cyth_funcname = cyth_helpers.get_cyth_name(funcname)
#        tweaked_example.source = replace_funcalls(example.source, funcname, cyth_funcname)
#        tweaked_examples.append(tweaked_example)
#    benchmark_iter = zip(doctest_examples, tweaked_examples)
    test_lines = []
    cyth_lines = []
    setup_lines = []
    cyth_funcname = cyth_helpers.get_cyth_name(funcname)
    for example in doctest_examples:
        if is_test(example.source, funcname):
            test_lines.append(example.source)
            cyth_lines.append(replace_funcalls(example.source, funcname, cyth_funcname))
        else:
            setup_lines.append(example.source)
    return list(zip(test_lines, cyth_lines)), utool.unindent(''.join(setup_lines))


#def run_benchmarks(funcname, docstring, iterations):
#    test_tuples, setup_script = make_benchmarks(funcname, docstring)
#    time_line = lambda line: timeit.timeit(stmt=line, setup=setup_script, number=iterations)
#    time_pair = lambda (x, y): (time_line(x), time_line(y))
#    return list(map(time_pair, test_tuples))


def emit_benchmark(funcname, docstring):
    test_tuples, setup_script = make_benchmarks(funcname, docstring)
    benchmark_name = utool.quasiquote('run_benchmark_{funcname}')
    #test_tuples, setup_script = make_benchmarks('''{funcname}''', '''{docstring}''')
    return (benchmark_name, utool.quasiquote("""
def {benchmark_name}(iterations):
    test_tuples = {test_tuples}
    setup_script = '''{setup_script}'''
    time_line = lambda line: timeit.timeit(stmt=line, setup=setup_script, number=iterations)
    time_pair = lambda (x, y): (time_line(x), time_line(y))
    def print_timing_info(tup):
        print(tup)
        (x, y) = time_pair(tup)
        print("Time for %d iterations of the python version: %d" % (iterations, x))
        print("Time for %d iterations of the cython version: %d" % (iterations, y))
        return (x, y)
    return list(map(print_timing_info, test_tuples))
"""))


class CythVisitor(BASE_CLASS):
    indent_level = 0
    emit = sys.stdout.write

    def __init__(self, indent_with=' ' * 4, add_line_information=False):
        super(CythVisitor, self).__init__(indent_with, add_line_information)
        self.benchmark_names = []
        self.benchmark_codes = []

    def get_result(self):
        return ''.join(self.result)

    def get_benchmarks(self):
        codes = '\n\n'.join(self.benchmark_codes)
        all_benchmarks = utool.indent('\n'.join([utool.quasiquote('{func}(iterations)') for func in self.benchmark_names]))
        return utool.quasiquote(utool.unindent("""
        #!/usr/bin/env python
        from __future__ import absolute_import, division, print_function
        import timeit
        {codes}

        def run_all_benchmarks(iterations):
        {all_benchmarks}

        if __name__ == '__main__':
            run_all_benchmarks(1000)
        """).strip())

    def process_args(self, args, vararg, kwarg, defaults=None):
        processed_argslist = map(self.visit, args)
        if vararg:
            processed_argslist.append('*%s' % vararg)
        if kwarg:
            processed_argslist.append('**%s' % kwarg)
        if defaults is not None:
            first_non_default = len(args) - len(defaults)
            for (ix, de) in enumerate(defaults):
                processed_argslist[first_non_default + ix] += '=%s' % self.visit(defaults[ix])
        return processed_argslist

    def signature(self, node, typedict={}):
        types_minus_sigtypes = typedict.copy()
        want_comma = []

        def write_comma():
            if want_comma:
                self.write(', ')
            else:
                want_comma.append(True)

        def loop_args(args, defaults):
            padding = [None] * (len(args) - len(defaults))
            for arg, default in zip(args, padding + defaults):
                if arg.id in typedict:
                    arg_ = typedict[arg.id] + ' ' + arg.id
                    types_minus_sigtypes.pop(arg.id)
                else:
                    arg_ = arg
                self.write(write_comma, arg_)
                self.conditional_write('=', default)

        loop_args(node.args, node.defaults)
        self.conditional_write(write_comma, '*', node.vararg)
        self.conditional_write(write_comma, '**', node.kwarg)

        kwonlyargs = getattr(node, 'kwonlyargs', None)
        if kwonlyargs:
            if node.vararg is None:
                self.write(write_comma, '*')
            loop_args(kwonlyargs, node.kw_defaults)
        return types_minus_sigtypes

    def parse_cythdef(self, cyth_def):
        """ Hacky string manipulation parsing """
        #ast.literal_eval(utool.unindent(cyth_def))

        typedict = {}
        cdef_mode = False
        current_indent = 0
        for line_ in cyth_def.splitlines():
            indent_level = utool.get_indentation(line_)
            # Check indentation
            if indent_level > current_indent:
                current_indent = indent_level
            elif indent_level > current_indent:
                current_indent = indent_level
                cdef_mode = False
            line = line_.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                continue
            # parse cdef
            if line.startswith('cdef:'):
                cdef_mode = True
                continue
            if cdef_mode or line.startswith('cdef '):
                assign_str = line.replace('cdef ', '')
                pos = assign_str.rfind(' ')
                type_ = assign_str[:pos]
                varstr = assign_str[(pos + 1):]
                typedict[varstr] = type_
        return typedict

    def typedict_to_cythdef(self, typedict):
        res = ['cdef:']
        for (id_, type_) in typedict.iteritems():
            #print("%s, %s" % (repr(id_), repr(type_)))
            res.append(self.indent_with + type_ + ' ' + id_)
        if len(typedict) == 0:
            res.append(self.indent_with + 'pass')
        return res

    def parse_cyth_markup(self, docstr, toplevel=False, funcname=None):
        comment_str = docstr.strip()
        doctest_examples = filter(lambda x: isinstance(x, doctest.Example),
                                    doctest.DocTestParser().parse(docstr))
        [print("doctest_examples[%d] = (%r, %r)" % (i, x, y)) for (i, (x, y)) in
            enumerate(map(lambda x: (x.source, x.want), doctest_examples))]
        has_markup = comment_str.find('<CYTH') != -1
        # type returned_action = [`defines of string * (string, string) Hashtbl.t | `replace of string] option
        tags_to_actions = [
            ('<CYTH>', lambda cyth_def: ('defines', cyth_def, self.parse_cythdef(cyth_def))),
            ('<CYTH:REPLACE>', lambda cyth_def: ('replace', utool.unindent(cyth_def))),
        ]
        end_tag = '</CYTH>'
        regex_flags = re.DOTALL | re.MULTILINE
        regex_to_actions = [(re.compile(tag + '(.*?)' + end_tag, regex_flags), act)
                            for tag, act in tags_to_actions]
        if has_markup:
            if funcname:
                (benchmark_name, benchmark_code) = emit_benchmark(funcname, docstr)
                self.benchmark_names.append(benchmark_name)
                self.benchmark_codes.append(benchmark_code)
            if toplevel:
                comment_str = re.sub('<CYTH>', '<CYTH:REPLACE>', comment_str)
            for (regex, action) in regex_to_actions:
                match = regex.search(comment_str)
                if match:
                    cyth_def = match.group(1)
                    return action(cyth_def)
            utool.printex(NotImplementedError('no known cyth tag in docstring'),
                           iswarning=True,
                           key_list=['comment_str'])
        return None

    def visit_Module(self, node):
        for subnode in node.body:
            if is_docstring(subnode):
                print('Encountered global docstring: %s' % repr(subnode.value.s))
                action = self.parse_cyth_markup(subnode.value.s, toplevel=True)
                if action:
                    if action[0] == 'replace':
                        cyth_def = action[1]
                        self.newline(extra=1)
                        self.write(cyth_def)
            elif isinstance(subnode, ast.FunctionDef):
                self.visit(subnode)
            else:
                #print('Skipping a global %r' % subnode.__class__)
                pass
        #return BASE_CLASS.visit_Module(self, node)

    def visit_FunctionDef(self, node):
        #super(CythVisitor, self).visit_FunctionDef(node)
        new_body = []
        cyth_action = None
        for stmt in node.body:
            if is_docstring(stmt):
                #print('found comment_str')
                docstr = stmt.value.s
                cyth_action = self.parse_cyth_markup(docstr, funcname=node.name)
            else:
                new_body.append(stmt)
        if cyth_action:
            if cyth_action[0] == 'defines':
                cyth_def = cyth_action[1]
                typedict = cyth_action[2]
                #self.decorators(node, 2)
                self.newline(extra=1)
                self.statement(node, 'def %s(' % node.name)
                types_minus_sigtypes = self.signature(node.args, typedict=typedict)
                cyth_def_body = self.typedict_to_cythdef(types_minus_sigtypes)
                self.write(')')
                if getattr(node, 'returns', None) is not None:
                    self.write(' ->', node.returns)
                self.write(':')
                self.indentation += 1
                for s in cyth_def_body:
                    self.write('\n', s)
                self.indentation -= 1
                self.body(new_body)
            elif cyth_action[0] == 'replace':
                cyth_def = cyth_action[1]
                self.newline(extra=1)
                self.write(cyth_def)

    #def visit_ClassDef(self, node):
    #    print(ast.dump(node))
    #    return BASE_CLASS.visit_ClassDef(self, node)

    #    processed_argslist = self.process_args(node.args.args, node.args.vararg, node.args.kwarg, node.args.defaults)
    #    output = "%sdef %s(%s):\n" % (self.prefix(), node.name, ', '.join(processed_argslist))
    #    self.indent()
    #    #print(codegen.to_source(node.body))
    #    for n in node.body:
    #        output += self.prefix() + self.visit(n) + '\n'
    #    self.dedent()
    #    return output

    #def prefix(self):
    #    return " " * self.indent_level

    #def indent(self):
    #    self.indent_level += 4

    #def dedent(self):
    #    self.indent_level -= 4

    #def visit_Module(self, node):
    #    return ''.join(map(self.visit, node.body))

    #def visit_Str(self, node):
    #    return repr(node.s)

    #def visit_Num(self, node):
    #    return str(node.n)

    #def visit_Expr(self, node):
    #    return self.visit(node.value)

    #def visit_Return(self, node):
    #    return ('return %s' % (self.visit(node.value)))

    #def visit_BinOp(self, node):
    #    return ('%s %s %s' % (self.visit(node.left), self.visit(node.op), self.visit(node.right)))

    #def visit_Name(self, node):
    #    return node.id

    #def visit_Add(self, node):
    #    return '+'

    #def visit_For(self, node):
    #    output = 'for %s in %s:\n' % (self.visit(node.target), self.visit(node.iter))
    #    self.indent()
    #    for n in node.body:
    #        output += '%s%s\n' % (self.prefix(), self.visit(n))
    #    self.dedent()
    #    return output

    #def visit_Call(self, node):
    #    output = '%s(%s)' % (self.visit(node.func), ', '.join(self.process_args(node.args, node.starargs, node.kwargs)))
    #    return output

    #def visit_AugAssign(self, node):
    #    return '%s %s= %s' % (self.visit(node.target), self.visit(node.op), self.visit(node.value))

    #def visit_Print(self, node):
    #    pass

    #def generic_visit(self, node):
    #    raise NotImplementedError("Not implemented for type %r" % (node.__class__))


#def cyth_process(pysource):
#    ast_root = ast.parse(pysource)
#
#
#def find_cyth_tags(py_text):
#    """
#    Parses between the <CYTH> </CYTH> tags. Tags must be the first or last
#    characters in the string so it doesn't pick up the ones in this docstr.
#    Also returns line numbers so future parsing is less intensive.
#    """
#    tagstr_list = []
#    lineno_list = []
#    return tagstr_list, lineno_list
#
#
#def parse_cythe_tags(tagstr_list, lineno_list, py_text):
#    """
#    creates new text for a pyx file
#    """
#    cython_text_blocks = []
#    cython_text = ''.join(cython_text_blocks)
#    return cython_text


def cythonize_fpath(py_fpath):
    print('[cyth] CYTHONIZE: py_fpath=%r' % py_fpath)
    cy_fpath = cyth_helpers.get_cyth_path(py_fpath)
    cy_bpath = cyth_helpers.get_cyth_bench_path(py_fpath)
    py_text = utool.read_from(py_fpath)
    visitor = CythVisitor()
    visitor.visit(ast.parse(py_text))
    cython_text = visitor.get_result()
    bench_text = visitor.get_benchmarks()
    utool.write_to(cy_fpath, cython_text)
    utool.write_to(cy_bpath, bench_text)


#cyth.import_cyth(__name__)

def translate(*paths):
    for fpath in paths:
        if isfile(fpath):
            abspath = utool.unixpath(fpath)
            cythonize_fpath(abspath)

if __name__ == '__main__':
    print('[cyth] main')
    input_path_list = utool.get_fpath_args(sys.argv[1:], pat='*.py')
    print(input_path_list)
    print('[cyth] nInput=%d' % (len(input_path_list,)))
    translate(*input_paths_list)
