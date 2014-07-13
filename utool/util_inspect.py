from __future__ import absolute_import, division, print_function
from . import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[alg]')


def list_class_funcnames(fname, blank_pats=['    #']):
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
            func_name = line[(def_x + 3):rparen_x]
            #print(func_name)
            funcname_list.append(func_name)
    return funcname_list
