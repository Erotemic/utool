# # -*- coding: utf-8 -*-
# """
# References:
#     http://code.activestate.com/recipes/576904/
#     http://code.activestate.com/recipes/277940-decorator-for-bindingconstants-at-compile-time/
# """
# from __future__ import print_function, division, absolute_import
# from opcode import opmap, HAVE_ARGUMENT, EXTENDED_ARG
# STORE_GLOBAL = opmap['STORE_GLOBAL']
# LOAD_GLOBAL  = opmap['LOAD_GLOBAL']
# LOAD_CONST   = opmap['LOAD_CONST']
# LOAD_ATTR    = opmap['LOAD_ATTR']
# BUILD_TUPLE  = opmap['BUILD_TUPLE']
# JUMP_FORWARD  = opmap['JUMP_FORWARD']
# #globals().update(opmap)


# def _make_constants(f, builtin_only=False, stoplist=[], verbose=False):
#     if verbose:
#         print('optimizing', f.__name__)
#     try:
#         co = f.__code__
#     except AttributeError:
#         return f        # Jython doesn't have a func_code attribute.
#     newcode = list(co.co_code)
#     newconsts = list(co.co_consts)
#     names = co.co_names
#     codelen = len(newcode)

#     #from six.moves import builtins
#     import six
#     if six.PY2:
#         import __builtin__
#         env = vars(__builtin__).copy()
#     else:
#         import builtins
#         env = vars(builtins).copy()
#     if builtin_only:
#         stoplist = dict.fromkeys(stoplist)
#         stoplist.update(f.__globals__)
#     else:
#         env.update(f.__globals__)

#     # First pass converts global lookups into constants
#     changed = False
#     i = 0
#     while i < codelen:
#         opcode = newcode[i]
#         if opcode in (EXTENDED_ARG, STORE_GLOBAL):
#             return f    # for simplicity, only optimize common cases
#         if opcode == LOAD_GLOBAL:
#             oparg = newcode[i + 1] + (newcode[i + 2] << 8)
#             name = co.co_names[oparg]
#             if name in env and name not in stoplist:
#                 value = env[name]
#                 for pos, v in enumerate(newconsts):
#                     if v is value:
#                         break
#                 else:
#                     pos = len(newconsts)
#                     newconsts.append(value)
#                 newcode[i] = LOAD_CONST
#                 newcode[i + 1] = pos & 0xFF
#                 newcode[i + 2] = pos >> 8
#                 changed = True
#                 if verbose:
#                     print(name, '-->', value)
#         i += 1
#         if opcode >= HAVE_ARGUMENT:
#             i += 2

#     # Second pass folds tuples of constants and constant attribute lookups
#     i = 0
#     while i < codelen:

#         newtuple = []
#         while newcode[i] == LOAD_CONST:
#             oparg = newcode[i + 1] + (newcode[i + 2] << 8)
#             newtuple.append(newconsts[oparg])
#             i += 3

#         opcode = newcode[i]
#         if not newtuple:
#             i += 1
#             if opcode >= HAVE_ARGUMENT:
#                 i += 2
#             continue

#         if opcode == LOAD_ATTR:
#             obj = newtuple[-1]
#             oparg = newcode[i + 1] + (newcode[i + 2] << 8)
#             name = names[oparg]
#             try:
#                 value = getattr(obj, name)
#             except AttributeError:
#                 continue
#             deletions = 1

#         elif opcode == BUILD_TUPLE:
#             oparg = newcode[i + 1] + (newcode[i + 2] << 8)
#             if oparg != len(newtuple):
#                 continue
#             deletions = len(newtuple)
#             value = tuple(newtuple)

#         else:
#             continue

#         reljump = deletions * 3
#         newcode[i - reljump] = JUMP_FORWARD
#         newcode[i - reljump + 1] = (reljump - 3) & 0xFF
#         newcode[i - reljump + 2] = (reljump - 3) >> 8

#         n = len(newconsts)
#         newconsts.append(value)
#         newcode[i] = LOAD_CONST
#         newcode[i + 1] = n & 0xFF
#         newcode[i + 2] = n >> 8
#         i += 3
#         changed = True
#         if verbose:
#             print("new folded constant:", value)

#     if not changed:
#         return f

#     codestr = bytes(newcode)
#     codeobj = type(co)(
#         co.co_argcount, co.co_kwonlyargcount, co.co_nlocals, co.co_stacksize,
#         co.co_flags, codestr, tuple(newconsts), co.co_names,
#         co.co_varnames, co.co_filename, co.co_name,
#         co.co_firstlineno, co.co_lnotab, co.co_freevars,
#         co.co_cellvars)
#     return type(f)(codeobj, f.__globals__, f.__name__, f.__defaults__,
#                     f.__closure__)

# _make_constants = _make_constants(_make_constants)  # optimize thyself!


# def bind_all(mc, builtin_only=False, stoplist=[],  verbose=False):
#     """Recursively apply constant binding to functions in a module or class.

#     Use as the last line of the module (after everything is defined, but
#     before test code).  In modules that need modifiable globals, set
#     builtin_only to True.

#     """
#     from types import FunctionType, ModuleType

#     def _bind_all(mc, builtin_only=False, stoplist=[],  verbose=False):
#         if isinstance(mc, (ModuleType, type)):
#             for k, v in list(vars(mc).items()):
#                 if type(v) is FunctionType:
#                     newv = _make_constants(v, builtin_only, stoplist,  verbose)
#                     setattr(mc, k, newv)
#                 elif isinstance(v, type):
#                     _bind_all(v, builtin_only, stoplist, verbose)

#     if isinstance(mc, dict):  # allow: bind_all(globals())
#         for k, v in list(mc.items()):
#             if type(v) is FunctionType:
#                 newv = _make_constants(v, builtin_only, stoplist,  verbose)
#                 mc[k] = newv
#             elif isinstance(v, type):
#                 _bind_all(v, builtin_only, stoplist, verbose)
#     else:
#         _bind_all(v, builtin_only, stoplist, verbose)


# @_make_constants
# def make_constants(builtin_only=False, stoplist=[], verbose=False):
#     """ Return a decorator for optimizing global references.

#     Replaces global references with their currently defined values.
#     If not defined, the dynamic (runtime) global lookup is left undisturbed.
#     If builtin_only is True, then only builtins are optimized.
#     Variable names in the stoplist are also left undisturbed.
#     Also, folds constant attr lookups and tuples of constants.
#     If verbose is True, prints each substitution as is occurs

#     """
#     if type(builtin_only) == type(make_constants):
#         raise ValueError("The bind_constants decorator must have arguments.")
#     return lambda f: _make_constants(f, builtin_only, stoplist, verbose)

# ## --------- Example call -----------------------------------------
# import random


# @make_constants(verbose=True, stoplist=['random'])
# def sample(population, k):
#     "Choose k unique random elements from a population sequence."

#     if not isinstance(population, (list, tuple, str)):
#         raise TypeError('Cannot handle type', type(population))
#     n = len(population)
#     if not 0 <= k <= n:
#         raise ValueError("sample larger than population")
#     result = [None] * k
#     pool = list(population)
#     for i in range(k):         # invariant:  non-selected at [0,n-i)
#         j = int(random.random() * (n - i))
#         result[i] = pool[j]
#         pool[j] = pool[n - i - 1]   # move non-selected item into vacancy
#     return result


# def sample_normal(population, k):
#     "Choose k unique random elements from a population sequence."

#     if not isinstance(population, (list, tuple, str)):
#         raise TypeError('Cannot handle type', type(population))
#     n = len(population)
#     if not 0 <= k <= n:
#         raise ValueError("sample larger than population")
#     result = [None] * k
#     pool = list(population)
#     for i in range(k):         # invariant:  non-selected at [0,n-i)
#         j = int(random.random() * (n - i))
#         result[i] = pool[j]
#         pool[j] = pool[n - i - 1]   # move non-selected item into vacancy
#     return result


# def test_example():
#     """
#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from utool.experimental.bytecode_optimizations import *
#         >>> from utool.experimental import bytecode_optimizations as bo
#         >>> dis1 = bo.get_disassembly_string(bo.sample)
#         >>> dis2 = bo.get_disassembly_string(bo.sample_normal)
#         >>> diff = ut.get_textdiff(dis1, dis2)
#         >>> print(diff)
#         >>> if len(diff) == 0:
#         ...    print('no difference')
#     """
#     import utool as ut

#     sample_optimized = make_constants(verbose=True, stoplist=['random'])(sample_normal)

#     opt_dis = get_disassembly_string(sample_optimized)
#     norm_dis = get_disassembly_string(sample_normal)
#     #norm2_dis = get_disassembly_string(sample)

#     # hmm  no difference
#     print(ut.get_textdiff(opt_dis, norm_dis))


# def get_disassembly_string(x, lasti=-1):
#     """
#     Reimplementation of disassemble that returns a string.
#     Disassemble a code object.

#     SeeAlso:
#         std library dis.py module
#     """
#     import opcode
#     import types
#     import dis
#     if x is None:
#         raise NotImplementedError('no default traceback dis')
#     if isinstance(x, types.InstanceType):
#         x = x.__class__
#     if hasattr(x, 'im_func'):
#         x = x.im_func
#     if hasattr(x, 'func_code'):
#         x = x.func_code
#     if hasattr(x, '__dict__'):
#         items = x.__dict__.items()
#         items.sort()
#         for name, x1 in items:
#             if isinstance(x1, dis._have_code):
#                 print("Disassembly of %s:" % name)
#                 try:
#                     dis(x1)
#                 except TypeError as msg:
#                     print("Sorry: %s" % msg)
#                 print('')
#     elif hasattr(x, 'co_code'):
#         # ok case
#         pass
#     elif isinstance(x, str):
#         raise NotImplementedError('no disassemble_string reimplement')
#     else:
#         raise TypeError('don\'t know how to disassemble %s objects' % type(x).__name__)
#     co = x
#     code = co.co_code
#     labels = dis.findlabels(code)
#     linestarts = dict(dis.findlinestarts(co))
#     n = len(code)
#     i = 0
#     extended_arg = 0
#     free = None
#     str_list = []
#     def appendstr(msg):
#         if msg is not None:
#             msg += ' '
#         else:
#             msg = '\n'
#         str_list.append(msg)

#     while i < n:
#         c = code[i]
#         op = ord(c)
#         if i in linestarts:
#             if i > 0:
#                 appendstr(None)
#             appendstr("%3d" % linestarts[i])
#         else:
#             appendstr('   ')

#         if i == lasti:
#             appendstr('-->')
#         else:
#             appendstr('   ')
#         if i in labels:
#             appendstr('>>')
#         else:
#             appendstr('  ')
#         appendstr(repr(i).rjust(4))
#         appendstr(opcode.opname[op].ljust(20))
#         i = i + 1
#         if op >= opcode.HAVE_ARGUMENT:
#             oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
#             extended_arg = 0
#             i = i + 2
#             if op == opcode.EXTENDED_ARG:
#                 extended_arg = oparg * 65536
#             appendstr(repr(oparg).rjust(5))
#             if op in opcode.hasconst:
#                 appendstr('(' + repr(co.co_consts[oparg]) + ')')
#             elif op in opcode.hasname:
#                 appendstr('(' + co.co_names[oparg] + ')')
#             elif op in opcode.hasjrel:
#                 appendstr('(to ' + repr(i + oparg) + ')')
#             elif op in opcode.haslocal:
#                 appendstr('(' + co.co_varnames[oparg] + ')')
#             elif op in opcode.hascompare:
#                 appendstr('(' + opcode.cmp_op[oparg] + ')')
#             elif op in opcode.hasfree:
#                 if free is None:
#                     free = co.co_cellvars + co.co_freevars
#                 appendstr('(' + free[oparg] + ')')
#         appendstr(None)
#     disassembly_string = ''.join(str_list)
#     #print(disassembly_string)
#     return disassembly_string


# """ Output from the example call:

# isinstance --> <built-in function isinstance>
# list --> <class 'list'>
# tuple --> <class 'tuple'>
# str --> <class 'str'>
# TypeError --> <class 'TypeError'>
# type --> <class 'type'>
# len --> <built-in function len>
# ValueError --> <class 'ValueError'>
# list --> <class 'list'>
# range --> <class 'range'>
# int --> <class 'int'>
# new folded constant: (<class 'list'>, <class 'tuple'>, <class 'str'>)
# """

# if __name__ == '__main__':
#     """
#     CommandLine:
#         python -m bytecode_optimizations
#         python -m bytecode_optimizations --allexamples
#         python -m bytecode_optimizations --allexamples --noface --nosrc
#     """
#     import multiprocessing
#     multiprocessing.freeze_support()  # for win32
#     import utool as ut  # NOQA
#     ut.doctest_funcs()
