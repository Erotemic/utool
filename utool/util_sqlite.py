# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from utool import util_inject
import six
import collections
print, rrr, profile = util_inject.inject2(__name__)


def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

SQLColumnRichInfo = collections.namedtuple('SQLColumnRichInfo', ('column_id', 'name', 'type_', 'notnull', 'dflt_value', 'pk'))


def get_table_columns(cur, tablename, exclude_columns=[]):
    import utool as ut
    colnames_ = ut.get_table_columnname_list(cur, tablename)
    colnames = tuple([colname for colname in colnames_ if colname not in exclude_columns])
    row_list = ut.get_table_rows(cur, tablename, colnames, unpack=False)
    column_list = zip(*row_list)
    return column_list


def get_table_csv(cur, tablename, exclude_columns=[]):
    """ Conveinience: Converts a tablename to csv format

    Args:
        tablename (str):
        exclude_columns (list):

    Returns:
        str: csv_table

    CommandLine:
        python -m ibeis.control.SQLDatabaseControl --test-get_table_csv

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.SQLDatabaseControl import *  # NOQA
        >>> # build test data
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> db = ibs.db
        >>> tablename = ibeis.const.NAME_TABLE
        >>> exclude_columns = []
        >>> # execute function
        >>> csv_table = db.get_table_csv(tablename, exclude_columns)
        >>> # verify results
        >>> result = str(csv_table)
        >>> print(result)
    """
    import utool as ut
    colnames_ = ut.get_table_columnname_list(cur, tablename)
    colnames = tuple([colname for colname in colnames_ if colname not in exclude_columns])
    row_list = ut.get_table_rows(cur, tablename, colnames, unpack=False)
    column_list = zip(*row_list)
    #=None, column_list=[], header='', column_type=None
    #import utool as ut
    #column_list, column_names = db.get_table_column_data(tablename, exclude_columns)
    # remove column prefix for more compact csvs
    column_lbls = [name.replace(tablename[:-1] + '_', '') for name in colnames]
    #header = db.get_table_csv_header(tablename)
    header = ''
    csv_table = ut.make_csv_table(column_list, column_lbls, header)
    return csv_table


def get_table_columnname_list(cur, tablename):
    colinfo_list_ = get_table_columninfo_list(cur, tablename)
    return [info[1] for info in colinfo_list_]


def get_table_columninfo_list(cur, tablename):
    """
    Args:
        tablename (str): table name

    Returns:
        column_list : list of tuples with format:
            (
                [0] column_id  : id of the column
                [1] name       : the name of the column
                [2] type_      : the type of the column (TEXT, INT, etc...)
                [3] notnull    : 0 or 1 if the column can contains null values
                [4] dflt_value : the default value
                [5] pk         : 0 or 1 if the column partecipate to the primary key
            )

    References:
        http://stackoverflow.com/questions/17717829/how-to-get-column-names-from-a-table-in-sqlite-via-pragma-net-c

    CommandLine:
        python -m utool.util_sqlite --test-get_table_columninfo_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_sqlite import *  # NOQA
    """
    cur.execute('PRAGMA TABLE_INFO("{tablename}")'.format(tablename=tablename))
    colinfo_list = cur.fetchall()
    colrichinfo_list = [SQLColumnRichInfo(*colinfo) for colinfo in colinfo_list]
    return colrichinfo_list


def get_primary_columninfo(cur, tablename):
    colinfo_list_ = get_table_columninfo_list(cur, tablename)
    colinfo_list = [colinfo for colinfo in colinfo_list_ if colinfo.pk]
    return colinfo_list


def get_nonprimary_columninfo(cur, tablename):
    colinfo_list_ = get_table_columninfo_list(cur, tablename)
    colinfo_list = [colinfo for colinfo in colinfo_list_ if not colinfo.pk]
    return colinfo_list


def get_table_num_rows(cur, tablename):
    cur.execute('SELECT COUNT(*) FROM {tablename}'.format(tablename=tablename))
    num_rows = cur.fetchall()[0][0]
    return num_rows


def get_table_column(cur, tablename, colname):
    """ Conveinience: """
    return get_table_rows(cur, tablename, colname)


def get_table_rows(cur, tablename, colnames, where=None, params=None, unpack=True):
    import utool as ut
    want_single_column = isinstance(colnames, six.string_types)
    want_single_param = params is not None and not ut.isiterable(params)
    #isinstance(params, six.string_types)
    if want_single_column:
        colnames = (colnames,)
    if colnames is not None and colnames != '*':
        assert isinstance(colnames, tuple), 'colnames must be a tuple'
        colnames_str = ', '.join(colnames)
    else:
        colnames_str = '*'
    #if isinstance(colnames, six.string_types):
    #    colnames = (colnames,)
    fmtdict = {
        'tablename'     : tablename,
        'colnames'    : colnames_str,
        'orderby'     : '',
    }

    #ORDER BY rowid ASC
    if where is None:
        operation_fmt = '''
        SELECT {colnames}
        FROM {tablename}
        {orderby}
        '''
    else:
        fmtdict['where_clause'] = where
        operation_fmt = '''
        SELECT {colnames}
        FROM {tablename}
        WHERE {where_clause}
        {orderby}
        '''
    operation_str = operation_fmt.format(**fmtdict)
    if params is None:
        cur.execute(operation_str)
        val_list = cur.fetchall()
    elif want_single_param:
        cur.execute(operation_str, (params,))
        val_list = cur.fetchall()
    else:
        # Execute many
        def executemany_scalar_generator(operation_str, params):
            for param in params:
                cur.execute(operation_str, param)
                vals = cur.fetchall()
                #assert len(vals) == 1, 'vals=%r, len(vals)=%r' % (vals, len(vals))
                yield vals
        val_list = list(executemany_scalar_generator(operation_str, params))

    if unpack:
        if want_single_column:
            # want a single value per parameter
            val_list = [val[0] for val in val_list]

        if want_single_param:
            # wants a single parameter
            assert len(val_list) == 1
            val_list = val_list[0]
    return val_list


def print_database_structure(cur):
    import utool as ut
    tablename_list = ut.get_tablenames(cur)
    colinfos_list = [ut.get_table_columninfo_list(cur, tablename) for tablename in tablename_list]
    numrows_list = [ut.get_table_num_rows(cur, tablename) for tablename in tablename_list]
    for tablename, colinfo_list, num_rows in ut.sortedby(list(zip(tablename_list, colinfos_list, numrows_list)), numrows_list):
        print('+-------------')
        print('tablename = %r' % (tablename,))
        print('num_rows = %r' % (num_rows,))
        #print(ut.repr4(colinfo_list))
        print(ut.repr4(ut.get_primary_columninfo(cur, tablename)))
        print(ut.repr4(ut.get_nonprimary_columninfo(cur, tablename)))
        print('+-------------')
