def test_jedi_can_read_googlestyle():
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


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/utool/utool/util_scripts
        python ~/code/utool/utool/util_scripts/test_jedi.py
    """
    test_jedi_can_read_googlestyle()
