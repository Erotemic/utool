#!/usr/env/python2.7
"""
ut
python cyth/cyth_script.py ~/code/fpath
cyth_script.py ~/code/ibeis/ibeis/model/hots
cyth_script.py "~/code/vtool/vtool"

"""
from __future__ import absolute_import, division, print_function
import utool
import sys
from os.path import splitext, isfile
import ast

#class CythTransformer(ast.NodeTransformer):
#    #

sample_code = """
#from __future__ import print_function
def foo(x, y):
        '''
        <CYTH>
        cdef:
            long x
            long y
        </CYTH>
        '''
        for i in range(42):
            x += 1
            print(x)
        return x + y
"""

class CythVisitor(ast.NodeVisitor):
    indent_level = 0
    emit = sys.stdout.write # instance variable to allow easily changing where it outputs to
#    suffixlist = ["ClassDef", "Return", "Delete", "Assign", "AugAssign", "Print", "For", "While", "If", "With", "Raise", "TryExcept", "TryFinally", "Assert", "Import", "ImportFrom", "Exec", "Global", "Expr", "Pass", "Break", "Continue"]
#    def __getattr__(self, key):
#        for suffix in [suffixlist]:
#            if key == visit + suffix:
#                return self.visit_genericthingy
#    def visit_genericthingy(self, node):
#        self.emit()
    def prefix(self):
        return " " * self.indent_level
    def indent(self):
        self.indent_level += 4
    def dedent(self):
        self.indent_level -= 4
#    def expr_to_string(self, node):
#        c = node.__class__
#        if c == ast.Str:
#            return node.s
#        elif c == ast.Num:
#            return str(node.expr)
#        elif c == ast.Expr:
#            return expr_to_string(c.value)
#        else:
#            assert False, "Unknown type in expr_to_string()"
    def visit_Module(self, node):
        return ''.join(map(self.visit, node.body))

    def process_args(self, args, vararg, kwarg, defaults=None):
        processed_argslist = map(self.visit, args)
        if vararg:
            processed_argslist.append('*%s' % vararg)
        if kwarg:
            processed_argslist.append('**%s' % kwarg)
        if defaults is not None:
            first_non_default = len(args) - len(defaults)
            for (i, de) in enumerate(defaults):
                processed_argslist[first_non_default+i] += '=%s' % self.visit(defaults[i])
        return processed_argslist

    def visit_FunctionDef(self, node):
        processed_argslist = self.process_args(node.args.args, node.args.vararg, node.args.kwarg, node.args.defaults)
        output = "%sdef %s(%s):\n" % (self.prefix(), node.name, ', '.join(processed_argslist))
        self.indent()
        for n in node.body:
            output += self.prefix() + self.visit(n) + '\n'
        self.dedent()
        return output

    def visit_Str(self, node):
        return repr(node.s)

    def visit_Num(self, node):
        return str(node.n)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Return(self, node):
        return ('return %s' % (self.visit(node.value)))

    def visit_BinOp(self, node):
        return ('%s %s %s' % (self.visit(node.left), self.visit(node.op), self.visit(node.right)))

    def visit_Name(self, node):
        return node.id

    def visit_Add(self, node):
        return '+'

    def visit_For(self, node):
        output = 'for %s in %s:\n' % (self.visit(node.target), self.visit(node.iter))
        self.indent()
        for n in node.body:
            output += '%s%s\n' % (self.prefix(), self.visit(n))
        self.dedent()
        return output

    def visit_Call(self, node):
        output = '%s(%s)' % (self.visit(node.func), ', '.join(self.process_args(node.args, node.starargs, node.kwargs)))
        return output

    def visit_AugAssign(self, node):
        return '%s %s= %s' % (self.visit(node.target), self.visit(node.op), self.visit(node.value))
#
#    def visit_Print(self, node):
#        pass

    def generic_visit(self, node):
        raise NotImplementedError("Not implemented for type %r" % (node.__class__))

def cyth_process(pysource):
    ast_root = ast.parse(pysource)
    #



def find_cyth_tags(py_text):
    """
    Parses between the <CYTHE> </CYTHE> tags. Tags must be the first or last
    characters in the string so it doesn't pick up the ones in this docstr.
    Also returns line numbers so future parsing is less intensive.
    """
    tagstr_list = []
    lineno_list = []
    return tagstr_list, lineno_list


def parse_cythe_tags(tagstr_list, lineno_list, py_text):
    """
    creates new text for a pyx file
    """
    cython_text_blocks = []
    cython_text = ''.join(cython_text_blocks)
    return cython_text


def cythonize_fpath(py_fpath):
    print('[cyth] CYTHONIZE: py_fpath=%r' % py_fpath)
    cy_fpath = splitext(py_fpath)[0] + '_cyth.pyx'
    py_text = utool.read_from(py_fpath)
    lineno_list, tagstr_list = find_cyth_tags(py_text)
    if len(tagstr_list) > 0:
        cython_text = parse_cythe_tags(tagstr_list, lineno_list, py_fpath)
        utool.write_to(cy_fpath, cython_text)


if __name__ == '__main__':
    print('[cyth] main')
    input_path_list = utool.get_fpath_args(sys.argv[1:], pat='*.py')
    print('[cyth] nInput=%d' % (len(input_path_list,)))
    for fpath in input_path_list:
        if isfile(fpath):
            abspath = utool.unixpath(fpath)
            cythonize_fpath(abspath)
