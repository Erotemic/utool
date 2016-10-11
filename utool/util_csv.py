# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
try:
    import numpy as np
except ImportError as ex:
    pass
from six.moves import zip, map
import six
from utool import util_type
from utool import util_inject
from utool import util_dev
print, rrr, profile = util_inject.inject2(__name__)


class CSV(util_dev.NiceRepr):

    def __init__(self, row_data, row_headers=None, col_headers=None):
        self.row_data = row_data
        if col_headers is None:
            self.header = row_data[0]
        else:
            self.header = col_headers
        self.header_tags = [[x] for x in self.header]
        self.short_header = None
        # FIXME: finish row/col header integration
        self.row_headers = row_headers

    def __nice__(self):
        import utool as ut
        if self.short_header is None:
            header_str =  ', '.join([ut.truncate_str(h, maxlen=15, truncmsg='~//~')
                                     for h in self.header])
        else:
            header_str =  ', '.join(self.short_header)
        return '(shape=%s: cols=%s)' % (self.shape, header_str,)

    @classmethod
    def from_fpath(cls, fpath):
        self = cls(read_csv(fpath))
        return self

    @property
    def shape(self):
        return len(self.row_data), len(self.header)

    def __str__(self):
        return self.nice_table()

    def _strip_self(self):
        self.row_data = [[c.strip(' ') for c in r] for r in self.row_data]
        self.header = self.row_data[0]
        self.header_tags = [[x] for x in self.header]

    def tabulate(self):
        import tabulate
        import utool as ut
        tabular_data = [ut.flatten([[r], d]) for r, d in zip(self.row_headers, self.row_data)]
        return tabulate.tabulate(tabular_data, [''] + self.header, 'fancy_grid')

    def transpose(self):
        import utool as ut
        row_dataT = ut.listT(self.row_data)
        return CSV(row_dataT, row_headers=self.header, col_headers=self.row_headers)

    def nice_table(self):
        import utool as ut
        return ut.make_csv_table(ut.listT(self.row_data), raw=True)

    def nice_table2(self, **kwargs):
        import utool as ut
        return ut.make_csv_table(ut.listT(self.row_data), column_lbls=self.header, row_lbls=self.row_headers, **kwargs)

    def raw_table(self):
        return '\n'.join([','.join([y for y in x]) for x in self.row_data])

    def fuzzy_filter_columns(self, fuzzy_headers):
        import utool as ut
        col_flags = ut.filterflags_general_tags(
            self.header_tags, logic='or',
            in_any=fuzzy_headers)
        self.header = ut.compress(self.header, col_flags)
        self.header_tags = ut.compress(self.header_tags, col_flags)
        self.row_data = ut.listT(ut.compress(ut.listT(self.row_data), col_flags))
        if self.short_header is not None:
            self.short_header = ut.compress(self.short_header, col_flags)

    def __getitem__(self, pat):
        colx = self.fuzzy_find_colx(pat)
        return self.take_column(colx)

    def fuzzy_reorder_columns(self, fuzzy_headers, inplace=True):
        import utool as ut
        specified_xs = [self.fuzzy_find_colx(pat) for pat in fuzzy_headers]
        otherxs = ut.index_complement(specified_xs, len(self.header_tags))
        new_order = specified_xs + otherxs
        return self.permute_columns(new_order)

    def permute_columns(self, new_order, inplace=True):
        import utool as ut
        self.header = ut.take(self.header, new_order)
        self.header_tags = ut.take(self.header_tags, new_order)
        self.row_data = ut.listT(ut.take(ut.listT(self.row_data), new_order))
        if self.short_header is not None:
            self.short_header = ut.take(self.short_header, new_order)
        return self

    def fuzzy_find_colxs(self, pat):
        import utool as ut
        colxs = ut.where(ut.filterflags_general_tags(self.header_tags, in_any=[pat]))
        return colxs

    def fuzzy_find_colx(self, pat):
        colxs = self.fuzzy_find_colxs(pat)
        assert len(colxs) == 1, ('cannot find column matching %r' % (pat,))
        return colxs[0]

    def take_fuzzy_column(self, pat):
        import utool as ut
        colx = self.fuzzy_find_colx(pat)
        self.take_column(colx)
        return ut.take_column(self.row_data, colx)

    def take_column(self, colx, with_header=True):
        import utool as ut
        if with_header:
            return ut.take_column(self.row_data, colx)
        else:
            return ut.take_column(self.row_data[1:], colx)

    def compress_rows(self, flags, with_header=True, inplace=True):
        if not inplace:
            import copy
            self = copy.deepcopy(self)
        import utool as ut
        if with_header:
            assert flags[0] is True
            self.row_data = ut.compress(self.row_data, flags)
        else:
            self.row_data = self.row_data[0:1] + ut.compress(self.row_data[1:], flags)
        return self

    def compress_cols(self, flags):
        pass


def numpy_to_csv(arr, col_lbls=None, header='', col_type=None):
    col_list = arr.T.tolist()
    return make_csv_table(col_list, col_lbls, header, col_type)


def read_csv(fpath):
    """ reads csv in unicode """
    import csv
    import utool as ut
    #csvfile = open(fpath, 'rb')
    with open(fpath, 'rb') as csvfile:
        row_iter = csv.reader(csvfile, delimiter=str(','), quotechar=str('|'))
        row_list = [ut.lmap(ut.ensure_unicode, row) for row in row_iter]
    return row_list


def make_standard_csv(column_list, column_lbls=None):
    from six.moves import cStringIO as StringIO
    import utool as ut
    import csv
    stream = StringIO()
    row_list = ut.listT(column_list)
    if six.PY2:
        row_list = [[ut.ensure_unicode(c).encode('utf-8')
                     for c in r]
                    for r in row_list]
        if column_lbls is not None:
            column_lbls = [ut.ensure_unicode(c).encode('utf-8')
                           for c in column_lbls]
    writer = csv.writer(stream, dialect=csv.excel)
    if column_lbls is not None:
        writer.writerow(column_lbls)
    writer.writerows(row_list)
    csv_str = stream.getvalue()
    return csv_str


def make_csv_table(column_list=[], column_lbls=None, header='',
                   column_type=None, row_lbls=None, transpose=False,
                   precision=2, use_lbl_width=True, comma_repl='<com>',
                   raw=False, new=False, standardize=False):
    """
    Creates a csv table with aligned columns

    make_csv_table

    Args:
        column_list (list):
        column_lbls (None):
        header (str):
        column_type (None):
        row_lbls (None):
        transpose (bool):

    Returns:
        str: csv_text

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_csv import *  # NOQA
        >>> column_list = [[1, 2, 3], ['A', 'B', 'C']]
        >>> column_lbls = ['num', 'alpha']
        >>> header = '# Test CSV'
        >>> column_type = (int, str)
        >>> row_lbls = None
        >>> transpose = False
        >>> csv_text = make_csv_table(column_list, column_lbls, header, column_type, row_lbls, transpose)
        >>> result = csv_text
        >>> print(result)
        # Test CSV
        # num_rows=3
        #   num,  alpha
              1,      A
              2,      B
              3,      C
    """
    import utool as ut

    assert comma_repl.find(',') == -1, 'comma_repl cannot contain a comma!'
    if transpose:
        column_lbls, row_lbls = row_lbls, column_lbls
        column_list = list(map(list, zip(*column_list)))
    if row_lbls is not None:
        if isinstance(column_list, np.ndarray):
            column_list = column_list.tolist()
        if isinstance(row_lbls, np.ndarray):
            row_lbls = row_lbls.tolist()
        column_list = [row_lbls] + column_list
        column_lbls = ['ROWLBL'] + list(map(six.text_type, column_lbls))
        if column_type is not None:
            column_type =  [six.text_type] + column_type
    if len(column_list) == 0:
        print('[csv] No columns')
        return header
    column_len = [len(col) for col in column_list]
    num_data = column_len[0]
    if num_data == 0:
        #print('[csv.make_csv_table()] No data. (header=%r)' % (header,))
        return header
    if any([num_data != clen for clen in column_len]):
        print('[csv] column_lbls = %r ' % (column_lbls,))
        print('[csv] column_len = %r ' % (column_len,))
        print('[csv] inconsistent column lengths')
        return header

    if column_type is None:
        column_type = list(map(type, ut.get_list_column(column_list, 0)))
        #column_type = [type(col[0]) for col in column_list]

    csv_rows = []
    if new:
        csv_rows.append(header)
    elif not raw:
        csv_rows.append(header)
        if not standardize:
            csv_rows.append('# num_rows=%r' % num_data)

    column_maxlen = []
    column_str_list = []

    if column_lbls is None:
        column_lbls = [''] * len(column_list)

    def _toint(c):
        if c is None:
            return 'None'
        try:
            if np.isnan(c):
                return 'nan'
        except TypeError as ex:
            print('------')
            print('[csv] TypeError %r ' % ex)
            print('[csv] _toint(c) failed')
            print('[csv] c = %r ' % c)
            print('[csv] type(c) = %r ' % type(c))
            print('------')
            raise
        return ('%d') % int(c)

    import uuid
    textable_types = [uuid.UUID, six.text_type]

    try:
        if standardize:
            def csv_format(r):
                text = ut.repr2(r, precision=precision)
                #text = six.text_type(r)
                # Check if needs escape
                escape_chars = ['"', ' ', ',']
                if any([c in text for c in escape_chars]):
                    # escape quotes with quotes
                    text = text.replace('"', '""')
                    # encapsulate with quotes
                    text = '"' + text + '"'
                return text
            for col, lbl, coltype in zip(column_list, column_lbls, column_type):
                col_str = [csv_format(r) for r in col]
                column_str_list.append(col_str)
                pass
        else:
            # Loop over every column
            for col, lbl, coltype in zip(column_list, column_lbls, column_type):
                # Loop over every row in the column (using list comprehension)
                if coltype is list or util_type.is_list(coltype):
                    col_str = [six.text_type(c).replace(',', ' ').replace('.', '<dot>')
                               for c in col]
                elif (coltype is float or
                      util_type.is_float(coltype) or
                      coltype == np.float32 or
                      util_type.is_valid_floattype(coltype)):
                    precision_fmtstr = '%.' + six.text_type(precision) + 'f'
                    col_str = ['None' if r is None else precision_fmtstr % float(r)
                               for r in col]
                    #col_ = [r if r is None else float(r) for r in col]
                    #col_str = [ut.repr2(r, precision=2) for r in col_]
                elif coltype is int or util_type.is_int(coltype) or coltype == np.int64:
                    col_str = [_toint(c) for c in (col)]
                elif coltype in textable_types or util_type.is_str(coltype):
                    col_str = [six.text_type(c).replace(',', comma_repl) for c in col]
                else:
                    print('[csv] is_unknown coltype=%r' % (coltype,))
                    try:
                        col_str = [six.text_type(c) for c in (col)]
                    except UnicodeDecodeError:
                        try:
                            col_str = [ut.ensure_unicode(c) for c in (col)]
                        except Exception:
                            col_str = [repr(c) for c in (col)]
                column_str_list.append(col_str)

        for col_str, lbl in zip(column_str_list, column_lbls):
            col_lens = [len(s) for s in (col_str)]
            max_len  = max(col_lens)
            if use_lbl_width:
                # The column label counts towards the column width
                max_len  = max(len(lbl), max_len)
            column_maxlen.append(max_len)
    except Exception as ex:
        #ut.embed()
        ut.printex(ex, keys=['col', 'lbl', 'coltype'])
        raise

    def _fmtfn(maxlen):
        return  ''.join(['%', six.text_type(maxlen + 2), 's'])
    fmtstr = ','.join([_fmtfn(maxlen) for maxlen in column_maxlen])
    try:
        if new:
            csv_rows.append('# ' + fmtstr % tuple(column_lbls))
        elif not raw:
            csv_rows.append('# ' + fmtstr % tuple(column_lbls))
            #csv_rows.append('# ' + fmtstr % column_lbls)
    except Exception as ex:
        #print(len(column_list))
        #ut.embed()
        ut.printex(ex, keys=['fmtstr', 'column_lbls'])
        raise
    for row in zip(*column_str_list):
        csv_rows.append('  ' + fmtstr % row)

    csv_text = '\n'.join(csv_rows)
    return csv_text

if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_csv; utool.doctest_funcs(utool.util_csv, allexamples=True)"
        python -c "import utool, utool.util_csv; utool.doctest_funcs(utool.util_csv)"
        python -m utool.util_csv
        python -m utool.util_csv --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
