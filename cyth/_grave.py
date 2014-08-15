
#from copy import deepcopy
#import timeit

#class CythTransformer(ast.NodeTransformer):
#    #

#BASE_CLASS = codegen.SourceGenerator
#BASE_CLASS = ast.NodeVisitor
#BASE_CLASS = astor.ExplicitNodeVisitor



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



    #print('<<<')
    #try:
    #    print(len(pt.body) == 1)
    #    print(isinstance(pt.body[0], ast.Expr))
    #    print(isinstance(pt.body[0].value, ast.Call))
    #    print(isinstance(pt.body[0].value.func, ast.Name))
    #    print(pt.body[0].value.func.id)
    #except:
    #    pass
    #print('>>>')



    #[print("doctest_examples[%d] = (%r, %r)" % (i, x, y)) for (i, (x, y)) in
    #    enumerate(map(lambda x: (x.source, x.want), doctest_examples))]
    #tweaked_examples = []
    ## would this be clearer with map?
    #for example in doctest_examples:
    #    tweaked_example = deepcopy(example)
    #    cyth_funcname = cyth_helpers.get_cyth_name(funcname)
    #    tweaked_example.source = replace_funcalls(example.source, funcname, cyth_funcname)
    #    tweaked_examples.append(tweaked_example)
    #benchmark_iter = zip(doctest_examples, tweaked_examples)



#def run_benchmarks(funcname, docstring, iterations):
#    test_tuples, setup_script = make_benchmarks(funcname, docstring)
#    time_line = lambda line: timeit.timeit(stmt=line, setup=setup_script, number=iterations)
#    time_pair = lambda (x, y): (time_line(x), time_line(y))
#    return list(map(time_pair, test_tuples))
