#!/usr/bin/python2.7
from __future__ import absolute_import, division, print_function
import ast
#import codegen
import astor  # NOQA;
from cyth_script import *  # NOQA

sample_code = """
#from __future__ import print_function
def foo(x, y):
        '''
        >>> foo = 'x'
        >>> foo
        x

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

pt = ast.parse(sample_code)
#print(astor.to_source(pt))
print('---------------------')
print('Abstract Syntax Tree:')
print(ast.dump(pt))
print('---------------------\n')

print('---------------------')
print('Cyth Visit Simple')

visitor = CythVisitor()
visitor.visit(pt)
print('---------------------')
print(visitor.get_result())


# More complicated test
import vtool
import utool
from os.path import join
py_fpath = join(vtool.__path__[0], 'keypoint.py')
py_text = utool.read_from(py_fpath)

parse_tree = ast.parse(py_text)
print(py_fpath)

print('---------------------')
#print('Abstract Syntax Tree 2:')
#print(ast.dump(parse_tree))

print('---------------------')
print('Cyth Visit Complex 2')
visitor2 = CythVisitor()
visitor2.visit(parse_tree)
print('---------------------')
print(visitor2.get_result())
print('---------------------')
print(visitor2.get_benchmarks())
print('---------------------')

#print(astor.to_source(parse_tree))
