from __future__ import absolute_import, division, print_function
import ast
from cyth_script import *
pt = ast.parse(sample_code)
print(ast.dump(pt))
a = CythVisitor()
print(a.visit(pt))

