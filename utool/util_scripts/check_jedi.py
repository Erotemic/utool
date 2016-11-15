def check_jedi_can_read_googlestyle():
    import jedi
    import utool as ut
    source1 = ut.codeblock(
        r'''
        # STARTBLOCK
        def spam(data):
            r"""
            Args:
                data (utool.ColumnLists): a column list objct
            """
            data.
        # ENDBLOCK
        '''
    )
    source2 = ut.codeblock(
        r'''
        # STARTBLOCK
        def spam(ibs, bar):
            r"""
            Args:
                ibs (ibeis.IBEISController): an object
            """
            import jedi
            jedi.n
            x = ''
            x.l
            ibs.d
            bar.d
        # ENDBLOCK
        '''
    )

    print('\n---testing jedi with utool.ColumnLists')
    self = script = jedi.Script(source1, line=7, column=8)  # NOQA
    completions = script.completions()  # NOQA
    print('completions = %r' % (completions,))
    vartype = script.goto_definitions()
    print('vartype = %r' % (vartype,))

    print('\n---testing jedi with ibeis.IBEISController')
    script = jedi.Script(source2, line=10)
    script.completions()
    # Find the variable type of argument
    self = script = jedi.Script(source2, line=11, column=7)  # NOQA
    completions = script.completions()  # NOQA
    print('completions = %r' % (completions,))
    vartype = script.goto_definitions()
    print('vartype = %r' % (vartype,))

    print('\n---testing jedi with undefined object bar')
    self = script = jedi.Script(source2, line=12, column=7)  # NOQA
    vartype = script.goto_definitions()  # NOQA
    print('vartype = %r' % (vartype,))
    vardefs = script.goto_assignments()  # NOQA
    print('vardefs = %r' % (vardefs,))
    # foodef, = jedi.names(source2)
    # foomems = foodef.defined_names()
    # xdef = foomems[2]


def _insource_jedi_vim_test(data, ibs):
    """
    If jedi-vim supports google style docstrings you should be able to
    autocomplete ColumnLists methods for `data`

    Args:
        data (utool.ColumnLists): a column list objct
        ibs (ibeis.IBEISController): an object
    """
    # TESTME: type a dot and tab. Hopefully autocomplete will happen.
    data
    ibs
    import utool as ut
    xdata = ut.ColumnLists()
    xdata
    import ibeis
    xibs = ibeis.IBEISController()
    xibs


def check_jedi_closures():
    import jedi
    import textwrap
    source = textwrap.dedent(
        r'''
        def foo(data):
            import matplotlib
            import matplotlib

            def get_default_edge_data(graph, edge):
                data = graph.get_edge_data(*edge)
                if data is None:
                    if len(edge) == 3 and edge[2] is not None:
                        data = graph.get_edge_data(edge[0], edge[1], int(edge[2]))
                    else:
                        data = graph.get_edge_data(edge[0], edge[1])
                if data is None:
                    data = {}
                return data

            import matplotlib
        '''
    )
    print('SOURCE: ')
    print(source)
    lines = source.split('\n')
    test_positions = []
    for row, line in enumerate(lines):
        print('line = %r' % (line,))
        pat = 'import '
        import_pos = line.find(pat)
        if import_pos >= 0:
            module_pos = import_pos + len(pat) + 1
            print('Adding to tests ' + line[module_pos:])
            column = module_pos
            test_positions.append((row, column))

    for row, column in test_positions:
        script = jedi.Script(source, line=3, column=12)
        definitions = script.goto_definitions()
        print('----------')
        print('Script@(%r, %r):' % (row, column))
        print(lines[row])
        print(' ' * column + '^')
        print('definitions = %r' % (definitions,))
        print('----------')

    # script = jedi.Script(source, line=4, column=12)
    # definitions = script.goto_definitions()
    # print('definitions = %r' % (definitions,))

    # script = jedi.Script(source, line=num_lines - 2, column=12)
    # definitions = script.goto_definitions()
    # print('definitions = %r' % (definitions,))


def check_jedi_utool():
    import jedi
    import textwrap
    source = textwrap.dedent(
        r'''
        import utool as ut
        ut.'''
    )
    script = jedi.Script(source)

    definitions = script.goto_definitions()
    print('definitions = %r' % (definitions,))

    completions = script.completions()  # NOQA
    print('completions = %r' % (completions,))

    vardefs = script.goto_assignments()  # NOQA
    print('vardefs = %r' % (vardefs,))


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/utool/utool/util_scripts
        python ~/code/utool/utool/util_scripts/check_jedi.py
    """
    # check_jedi_can_read_googlestyle()
    # check_jedi_closures()
    check_jedi_utool()
