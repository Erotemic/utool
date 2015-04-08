from __future__ import absolute_import, division, print_function
try:
    import numpy as np
except ImportError as ex:
    pass
from six.moves import zip, map
from utool import util_type
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[csv]')


def numpy_to_csv(arr, col_lbls=None, header='', col_type=None):
    col_list = arr.T.tolist()
    return make_csv_table(col_list, col_lbls, header, col_type)


def make_csv_table(column_list=[], column_lbls=None, header='',
                   column_type=None, row_lbls=None, transpose=False,
                   precision=2, use_lbl_width=True, comma_repl='<comma>'):
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
        #import utool as ut
        #ut.embed()
        #column_lbls = row_lbls[0]
        #row_list =
    if row_lbls is not None:
        if isinstance(column_list, np.ndarray):
            column_list = column_list.tolist()
        if isinstance(row_lbls, np.ndarray):
            row_lbls = row_lbls.tolist()
        column_list = [row_lbls] + column_list
        column_lbls = ['ROWLBL'] + list(map(str, column_lbls))
        if column_type is not None:
            column_type =  [str] + column_type
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
    csv_rows.append(header)
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

    try:
        # Loop over every column
        for col, lbl, coltype in zip(column_list, column_lbls, column_type):
            # Loop over every row in the column (using list comprehension)
            if coltype is list or util_type.is_list(coltype):
                #print('list')
                #col_str = [str(c).replace(',', comma_repl).replace('.', '<dot>') for c in iter(col)]
                col_str = [str(c).replace(',', ' ').replace('.', '<dot>') for c in col]
            elif (coltype is float or
                  util_type.is_float(coltype) or
                  coltype == np.float32 or
                  util_type.is_valid_floattype(coltype)):
                precision_fmtstr = '%.' + str(precision) + 'f'
                col_str = ['None' if r is None else precision_fmtstr % float(r) for r in col]
            elif coltype is int or util_type.is_int(coltype) or coltype == np.int64:
                col_str = [_toint(c) for c in iter(col)]
            elif coltype is str or coltype is unicode or  util_type.is_str(coltype):
                col_str = [str(c).replace(',', comma_repl) for c in col]
            else:
                print('[csv] is_unknown coltype=%r' % (coltype,))
                col_str = [str(c) for c in iter(col)]
            col_lens = [len(s) for s in iter(col_str)]
            max_len  = max(col_lens)
            if use_lbl_width:
                # The column label counts towards the column width
                max_len  = max(len(lbl), max_len)
            column_maxlen.append(max_len)
            column_str_list.append(col_str)
    except Exception as ex:
        #ut.embed()
        ut.printex(ex, keys=['col', 'lbl', 'coltype'])
        raise

    _fmtfn = lambda maxlen: ''.join(['%', str(maxlen + 2), 's'])
    fmtstr = ','.join([_fmtfn(maxlen) for maxlen in column_maxlen])
    try:
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
