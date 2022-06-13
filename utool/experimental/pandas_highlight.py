from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
# print, rrr, profile = ut.inject2(__name__)


def monkey_to_str_columns(self, latex=False):
    import numpy as np
    import pandas as pd
    import utool as ut
    frame = self.tr_frame
    highlight_func = 'max'
    highlight_cols = self.highlight_cols

    highlight_func = np.argmax
    # lets us highlight more than one

    highlight_func = ut.partial(ut.argmax, multi=True)
    try:
        colvalues = frame[highlight_cols].values
    except Exception:
        colvalues = frame.values[:, highlight_cols]

    perrow_colxs = [highlight_func(row) for row in colvalues]

    n_rows = len(colvalues)
    n_cols = len(highlight_cols)
    shape = (n_rows, n_cols)
    multi_index = [(rx, cx) for rx, cxs in enumerate(perrow_colxs) for cx in cxs]
    flat_idxs = np.ravel_multi_index(list(zip(*multi_index)), shape)
    flags2d = np.zeros(shape, dtype=np.int32)
    flags2d.ravel()[flat_idxs] = 1
    # np.unravel_index(flat_idxs, shape)

    def color_func(val, level):
        if level:
            if latex:
                n_indent = ut.get_indentation(val)
                newval = (' ' * n_indent) + '\\textbf{' + val.lstrip(' ') + '}'
                return newval
            else:
                return ut.color_text(val, 'red')
        else:
            return val

    try:
        _make_fixed_width = pd.formats.format._make_fixed_width
    except AttributeError:
        _make_fixed_width = pd.io.formats.format._make_fixed_width

    frame = self.tr_frame
    str_index = self._get_formatted_index(frame)
    str_columns = self._get_formatted_column_labels(frame)
    if self.header:
        stringified = []
        for i, c in enumerate(frame):
            cheader = str_columns[i]
            max_colwidth = max(self.col_space or 0, *(self.adj.len(x)
                                                      for x in cheader))
            fmt_values = self._format_col(i)
            fmt_values = _make_fixed_width(fmt_values, self.justify,
                                           minimum=max_colwidth,
                                           adj=self.adj)
            max_len = max(np.max([self.adj.len(x) for x in fmt_values]),
                          max_colwidth)
            cheader = self.adj.justify(cheader, max_len, mode=self.justify)

            # Apply custom coloring
            # cflags = flags2d.T[i]
            # fmt_values = [color_func(val, level) for val, level in zip(fmt_values, cflags)]

            stringified.append(cheader + fmt_values)
    else:
        stringified = []
        for i, c in enumerate(frame):
            fmt_values = self._format_col(i)
            fmt_values = _make_fixed_width(fmt_values, self.justify,
                                           minimum=(self.col_space or 0),
                                           adj=self.adj)

            stringified.append(fmt_values)

    strcols = stringified
    if self.index:
        strcols.insert(0, str_index)

    # Add ... to signal truncated
    truncate_h = self.truncate_h
    truncate_v = self.truncate_v

    if truncate_h:
        col_num = self.tr_col_num
        # infer from column header
        col_width = self.adj.len(strcols[self.tr_size_col][0])
        strcols.insert(self.tr_col_num + 1, ['...'.center(col_width)] *
                       (len(str_index)))
    if truncate_v:
        n_header_rows = len(str_index) - len(frame)
        row_num = self.tr_row_num
        for ix, col in enumerate(strcols):
            # infer from above row
            cwidth = self.adj.len(strcols[ix][row_num])
            is_dot_col = False
            if truncate_h:
                is_dot_col = ix == col_num + 1
            if cwidth > 3 or is_dot_col:
                my_str = '...'
            else:
                my_str = '..'

            if ix == 0:
                dot_mode = 'left'
            elif is_dot_col:
                cwidth = self.adj.len(strcols[self.tr_size_col][0])
                dot_mode = 'center'
            else:
                dot_mode = 'right'
            dot_str = self.adj.justify([my_str], cwidth, mode=dot_mode)[0]
            strcols[ix].insert(row_num + n_header_rows, dot_str)

    for cx_ in highlight_cols:
        cx = cx_ + bool(self.header)
        col = strcols[cx]
        # Offset for the column header and possible index name
        base = int(self.has_index_names) + 1
        for rx_, val in enumerate(col[base:]):
            rx = rx_ + base
            strcols[cx][rx] = color_func(val, flags2d[rx_, cx_])

    return strcols


def to_string_monkey(df, highlight_cols=None, latex=False):
    """  monkey patch to pandas to highlight the maximum value in specified
    cols of a row

    Ignore:
        >>> from utool.experimental.pandas_highlight import *
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        >>>     np.array([[ 0.9,         0.86886931,  0.86842073,  0.9       ],
        >>>               [ 0.34196218,  0.34289191,  0.34206377,  0.34252863],
        >>>               [ 0.34827074,  0.34827074,  0.34827074,  0.34827074],
        >>>               [ 0.76979453,  0.77214855,  0.77547518,  0.38850962]]),
        >>>     columns=['sum(fgweights)', 'sum(weighted_ratio)', 'len(matches)', 'score_lnbnn_1vM'],
        >>>     index=['match_state(match-v-rest)', 'match_state(nomatch-v-rest)', 'match_state(notcomp-v-rest)', 'photobomb_state']
        >>> )
        >>> highlight_cols = 'all'
        >>> print(to_string_monkey(df, highlight_cols))
        >>> print(to_string_monkey(df, highlight_cols, latex=True))

    ut.editfile(pd.io.formats.printing.adjoin)
    """
    try:
        import pandas as pd
        import utool as ut
        import numpy as np
        import six
        if isinstance(highlight_cols, six.string_types) and highlight_cols == 'all':
            highlight_cols = np.arange(len(df.columns))
        # kwds = dict(buf=None, columns=None, col_space=None, header=True,
        #             index=True, na_rep='NaN', formatters=None,
        #             float_format=None, sparsify=None, index_names=True,
        #             justify=None, line_width=None, max_rows=None,
        #             max_cols=None, show_dimensions=False)
        # self = pd.formats.format.DataFrameFormatter(df, **kwds)
        try:
            self = pd.formats.format.DataFrameFormatter(df)
        except AttributeError:
            self = pd.io.formats.format.DataFrameFormatter(df)

        self.highlight_cols = highlight_cols

        def monkey(self):
            return monkey_to_str_columns(self, latex=latex)

        ut.inject_func_as_method(self, monkey, '_to_str_columns', override=True, force=True)

        def strip_ansi(text):
            import re
            ansi_escape = re.compile(r'\x1b[^m]*m')
            return ansi_escape.sub('', text)

        def justify_ansi(self, texts, max_len, mode='right'):
            if mode == 'left':
                return [x.ljust(max_len + (len(x) - len(strip_ansi(x)))) for x in texts]
            elif mode == 'center':
                return [x.center(max_len + (len(x) - len(strip_ansi(x)))) for x in texts]
            else:
                return [x.rjust(max_len + (len(x) - len(strip_ansi(x)))) for x in texts]
        ut.inject_func_as_method(self.adj, justify_ansi, 'justify', override=True, force=True)

        def strlen_ansii(self, text):
            return pd.compat.strlen(strip_ansi(text), encoding=self.encoding)
        ut.inject_func_as_method(self.adj, strlen_ansii, 'len', override=True, force=True)

        if False:
            strlen = ut.partial(strlen_ansii, self.adj)  # NOQA
            justfunc = ut.partial(justify_ansi, self.adj)  # NOQA
            # Essentially what to_string does
            strcols = monkey_to_str_columns(self)
            # texts = strcols[2]
            space = 1
            lists = strcols
            str_ = self.adj.adjoin(space, *lists)
            print(str_)
            print(strip_ansi(str_))
        self.to_string()
        result = self.buf.getvalue()
        # hack because adjoin is not working correctly with injected strlen
        result = '\n'.join([x.rstrip() for x in result.split('\n')])
        return result
    except Exception as ex:
        ut.printex('pandas monkey-patch is broken: {}'.format(str(ex)),
                   tb=True, iswarning=True)
        return str(df)


def pandas_repr(df):
    import utool as ut
    args = [
        df.values,
    ]
    kwargs = [
        ('columns', df.columns.values.tolist()),
        ('index', df.index.values.tolist()),
    ]
    header = 'pd.DataFrame('
    footer = ')'

    arg_parts = [
        ut.hz_str('    ', ut.repr2(arg))
        for arg in args if arg is not None
    ]
    kwarg_parts = [
        ut.hz_str('    {}={}'.format(key, ut.repr2(val)))
        for key, val in kwargs if val is not None
    ]
    body = ',\n'.join(arg_parts + kwarg_parts)
    dfrepr = '\n'.join([header, body, footer])
    print(dfrepr)
    pass
