# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from utool import util_inject
import six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[sqlite]')


def get_tablenames(cur):
    """ Conveinience: """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablename_list_ = cur.fetchall()
    tablename_list = [str(tablename[0]) for tablename in tablename_list_ ]
    return tablename_list

import collections
SQLColumnRichInfo = collections.namedtuple('SQLColumnRichInfo', ('column_id', 'name', 'type_', 'notnull', 'dflt_value', 'pk'))


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


def get_table_rows(cur, tablename, colnames, where=None, params=None, **kwargs):
    want_single_column = isinstance(colnames, six.string_types)
    if want_single_column:
        colnames = (colnames,)
    assert isinstance(colnames, tuple), 'colnames must be a tuple'
    #if isinstance(colnames, six.string_types):
    #    colnames = (colnames,)
    fmtdict = {'tablename'     : tablename,
               'colnames'    : ', '.join(colnames), }
    if where is None:
        operation_fmt = '''
        SELECT {colnames}
        FROM {tablename}
        '''
    else:
        fmtdict['where_clause'] = where
        operation_fmt = '''
        SELECT {colnames}
        FROM {tablename}
        WHERE {where_clause}
        '''
    operation_str = operation_fmt.format(**fmtdict)
    if params is None:
        cur.execute(operation_str)
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
    if want_single_column:
        val_list = [val[0] for val in val_list]
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
        #print(ut.list_str(colinfo_list))
        print(ut.list_str(ut.get_primary_columninfo(cur, tablename)))
        print(ut.list_str(ut.get_nonprimary_columninfo(cur, tablename)))
        print('+-------------')
