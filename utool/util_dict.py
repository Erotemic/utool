from __future__ import absolute_import, division, print_function
from itertools import product as iprod
from six.moves import zip
from collections import defaultdict, namedtuple
from collections import OrderedDict
from utool import util_inject
from utool import util_class
import six
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    pass
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[dict]')


DimensionBasis = namedtuple('DimensionBasis', ('dimension_name', 'dimension_point_list'))


def testdata_grid_search():
    import utool as ut
    grid_basis = [
        ut.util_dict.DimensionBasis('p', [.5, .6, .7, .8, .9, 1.0]),
        ut.util_dict.DimensionBasis('K', [2, 3, 4, 5]),
        ut.util_dict.DimensionBasis('clip_fraction', [.1, .2, .5, 1.0]),
    ]
    grid_searcher = ut.GridSearch(grid_basis)
    for cfgdict in grid_searcher:
        tp_score = cfgdict['p'] + (cfgdict['K'] ** .5)
        tn_score = (cfgdict['p'] * (cfgdict['K'])) / cfgdict['clip_fraction']
        grid_searcher.append_result(tp_score, tn_score)
    return grid_searcher


@six.add_metaclass(util_class.ReloadingMetaclass)
class GridSearch(object):
    """
    helper for executing iterations and analyzing the results of a grid search
    """
    def __init__(grid_searcher, grid_basis):
        grid_searcher.grid_basis = grid_basis
        grid_searcher.tp_score_list = []
        grid_searcher.tn_score_list = []
        grid_searcher.score_diff_list = []
        cfgdict_iter = grid_search_generator(grid_basis)
        grid_searcher.cfgdict_list = list(cfgdict_iter)

    def __iter__(grid_searcher):
        for cfgdict in grid_searcher.cfgdict_list:
            yield cfgdict

    def append_result(grid_searcher, tp_score, tn_score):
        diff = tp_score - tn_score
        grid_searcher.score_diff_list.append(diff)
        grid_searcher.tp_score_list.append(tp_score)
        grid_searcher.tn_score_list.append(tn_score)

    def get_score_list_and_lbls(grid_searcher):
        score_list  = [grid_searcher.score_diff_list,
                       grid_searcher.tp_score_list,
                       grid_searcher.tn_score_list]
        score_lbls  = ['score_diff', 'tp_score', 'tn_score']
        return score_list, score_lbls

    def get_param_list_and_lbls(grid_searcher):
        import utool as ut
        param_name_list = ut.get_list_column(grid_searcher.grid_basis, 0)
        params_vals = [list(six.itervalues(dict_)) for dict_ in grid_searcher.cfgdict_list]
        param_vals_list = list(zip(*params_vals))
        return param_name_list, param_vals_list

    def get_csv_results(grid_searcher, max_lines=None):
        import utool as ut
        # Input Parameters
        param_name_list, param_vals_list = grid_searcher.get_param_list_and_lbls()
        # Result Scores
        score_list, score_lbls = grid_searcher.get_score_list_and_lbls()

        score_lbl  = 'score_diff'
        score_vals = score_list[score_lbls.index(score_lbl)]
        sortby_func = ut.make_sortby_func(score_vals, reverse=True)

        score_name_sorted = score_lbls
        param_name_sorted = param_name_list
        score_list_sorted = list(map(sortby_func, score_list))
        param_vals_sorted = list(map(sortby_func, param_vals_list))

        # Build CSV
        column_lbls = score_name_sorted + param_name_sorted
        column_list = score_list_sorted + param_vals_sorted

        if max_lines is not None:
            column_list = [ut.listclip(col, max_lines) for col in column_list]
        header_raw_fmtstr = ut.codeblock(
            '''
            import utool as ut
            from utool import DimensionBasis
            title = 'Grid Search Results CSV'
            grid_basis = {grid_basis_str}
            ''')
        fmtdict = dict(
            grid_basis_str=ut.list_str(grid_searcher.grid_basis),
        )
        header_raw = header_raw_fmtstr.format(**fmtdict)
        header = ut.indent(header_raw, '# >>> ')
        #ut.rrrr()
        precision = 3
        csvtext = ut.make_csv_table(column_list, column_lbls, header, precision=precision)
        return csvtext

    def get_dimension_stats(grid_searcher, param_lbl, score_lbl='score_diff'):
        import utool as ut
        score_list, score_lbls = grid_searcher.get_score_list_and_lbls()
        param_name_list, param_vals_list = grid_searcher.get_param_list_and_lbls()
        param_vals = param_vals_list[param_name_list.index(param_lbl)]
        score_vals = score_list[score_lbls.index(score_lbl)]
        #sortby_func = ut.make_sortby_func(score_vals, reverse=True)
        #build_conflict_dict(param_vals, score_vals)
        param2_scores = ut.group_items(score_vals, param_vals)
        param2_score_stats = {
            param: ut.get_stats(scores)
            for param, scores in six.iteritems(param2_scores)
        }
        #print(ut.dict_str(param2_score_stats))
        return param2_score_stats

    def plot_dimension(grid_searcher, param_lbl, score_lbl='score_diff',
                       **kwargs):
        r"""
        Args:
            param_lbl (?):
            score_lbl (str):

        CommandLine:
            python -m utool.util_dict --test-plot_dimension
            python -m utool.util_dict --test-plot_dimension --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_dict import *  # NOQA
            >>> import plottool as pt
            >>> # build test data
            >>> grid_searcher = testdata_grid_search()
            >>> param_lbl = 'p'
            >>> score_lbl = 'score_diff'
            >>> grid_searcher.plot_dimension('p', score_lbl, fnum=1)
            >>> grid_searcher.plot_dimension('K', score_lbl, fnum=2)
            >>> grid_searcher.plot_dimension('clip_fraction', score_lbl, fnum=3)
            >>> pt.show_if_requested()
        """
        import plottool as pt
        param2_score_stats = grid_searcher.get_dimension_stats(param_lbl, score_lbl)
        title = param_lbl + ' vs ' + score_lbl
        fig = pt.interval_stats_plot(param2_score_stats, x_label=param_lbl,
                                     y_label=score_lbl, title=title, **kwargs)
        return fig


def count_dict_vals(dict_of_lists):
    count_dict = {'len(%s)' % (key,): len(val) for key, val in six.iteritems(dict_of_lists)}
    return count_dict


def get_dict_hashid(dict_):
    r"""
    Args:
        dict_ (?):

    Returns:
        ?: inverted_dict

    CommandLine:
        python -m utool.util_dict --test-get_dict_hashid

    References:
        http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> # build test data
        >>> dict_ = {}
        >>> dict_ = {'a': 'b'}
        >>> dict_ = {'a': {'c': 'd'}}
        >>> # execute function
        >>> hashid = get_dict_hashid(dict_)
        >>> # verify results
        >>> result = str(hashid)
        >>> print(result)
    """
    from utool import util_hash
    #hashid = hash(frozenset(dict_.items()))
    hashid = util_hash.make_hash(dict_)
    return hashid


def invert_dict(dict_):
    """
    invert_dict

    Args:
        dict_ (dict_):

    Returns:
        dict: inverted_dict

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {'a': 1, 'b': 2}
        >>> inverted_dict = invert_dict(dict_)
        >>> result = inverted_dict
        >>> print(result)
        {1: 'a', 2: 'b'}
    """
    inverted_dict = {val: key for key, val in six.iteritems(dict_)}
    return inverted_dict


def iter_all_dict_combinations_ordered(varied_dict):
    """
    Same as all_dict_combinations but preserves order
    """
    tups_list = [[(key, val) for val in val_list]
                 for (key, val_list) in six.iteritems(varied_dict)]
    dict_iter = (OrderedDict(tups) for tups in iprod(*tups_list))
    return dict_iter


def all_dict_combinations_ordered(varied_dict):
    """
    Same as all_dict_combinations but preserves order
    """
    dict_list = list(iter_all_dict_combinations_ordered)
    return dict_list


def grid_search_generator(grid_basis=[], *args, **kwargs):
    r"""
    Iteratively yeilds individual configuration points
    inside a defined basis.

    Args:
        grid_basis (list): a list of 2-component tuple. The named tuple looks
            like this:

    CommandLine:
        python -m utool.util_dict --test-grid_search_generator

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> grid_basis = [
        ... DimensionBasis('dim1', [.1, .2, .3]),
        ... DimensionBasis('dim2', [.1, .4, .5]),
        ... ]
        >>> args = tuple()
        >>> kwargs = {}
        >>> # execute function
        >>> point_list = list(grid_search_generator(grid_basis))
        >>> # verify results
        >>> column_lbls = ut.get_list_column(grid_basis, 0)
        >>> column_list  = ut.get_list_column(grid_basis, 1)
        >>> first_vals = ut.get_list_column(ut.get_list_column(grid_basis, 1), 0)
        >>> column_types = list(map(type, first_vals))
        >>> header = 'grid search'
        >>> result = ut.make_csv_table(column_list, column_lbls, header, column_types)
        >>> print(result)
        grid search
        # NumData 3
        #   dim1,  dim2
            0.10,  0.10
            0.20,  0.40
            0.30,  0.50

    """
    grid_basis_ = grid_basis + list(args) + list(kwargs.items())
    grid_basis_dict = OrderedDict(grid_basis_)
    grid_point_iter = iter_all_dict_combinations_ordered(grid_basis_dict)
    for grid_point in grid_point_iter:
        yield grid_point


def all_dict_combinations(varied_dict):
    """
    all_dict_combinations

    Args:
        varied_dict (dict):  a dict with lists of possible parameter settings

    Returns:
        list: dict_list a list of dicts correpsonding to all combinations of params settings

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> dict_list = all_dict_combinations(varied_dict)
        >>> result = str(ut.list_str(dict_list))
        >>> print(result)
        [
            {'pipeline_root': 'vsmany', 'sv_on': True, 'logdist_weight': 0.0},
            {'pipeline_root': 'vsmany', 'sv_on': True, 'logdist_weight': 1.0},
            {'pipeline_root': 'vsmany', 'sv_on': False, 'logdist_weight': 0.0},
            {'pipeline_root': 'vsmany', 'sv_on': False, 'logdist_weight': 1.0},
            {'pipeline_root': 'vsmany', 'sv_on': None, 'logdist_weight': 0.0},
            {'pipeline_root': 'vsmany', 'sv_on': None, 'logdist_weight': 1.0},
        ]
    """
    tups_list = [[(key, val) for val in val_list]
                 for (key, val_list) in six.iteritems(varied_dict)]
    dict_list = [dict(tups) for tups in iprod(*tups_list)]
    #dict_list = [{key: val for (key, val) in tups} for tups in iprod(*tups_list)]
    #from collections import OrderedDict
    #dict_list = [OrderedDict([(key, val) for (key, val) in tups]) for tups in iprod(*tups_list)]
    return dict_list


def all_dict_combinations_lbls(varied_dict):
    """ returns a label for each variation in a varydict.
    It tries to not be oververbose and returns only what parameters are varied
    in each label.

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> comb_lbls = utool.all_dict_combinations_lbls(varied_dict)
        >>> result = (utool.list_str(comb_lbls))
        >>> print(result)
        [
            (('sv_on', True), ('logdist_weight', 0.0)),
            (('sv_on', True), ('logdist_weight', 1.0)),
            (('sv_on', False), ('logdist_weight', 0.0)),
            (('sv_on', False), ('logdist_weight', 1.0)),
            (('sv_on', None), ('logdist_weight', 0.0)),
            (('sv_on', None), ('logdist_weight', 1.0)),
        ]

    """
    multitups_list = [[(key, val) for val in val_list]
                      for key, val_list in six.iteritems(varied_dict)
                      if len(val_list) > 1]
    comb_lbls = list(map(str, list(iprod(*multitups_list))))
    return comb_lbls


def dict_union2(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def dict_union(*args):
    return dict([item for dict_ in iter(args) for item in six.iteritems(dict_)])


def items_sorted_by_value(dict_):
    sorted_items = sorted(six.iteritems(dict_), key=lambda k, v: v[1])
    return sorted_items


def keys_sorted_by_value(dict_):
    sorted_keys = sorted(dict_, key=lambda key: dict_[key])
    return sorted_keys


def build_conflict_dict(key_list, val_list):
    """
    Builds dict where a list of values is associated with more than one key

    Args:
        key_list (list):
        val_list (list):

    Returns:
        dict: key_to_vals

    CommandLine:
        python -m utool.util_dict --test-build_conflict_dict

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> key_list = [  1,   2,   2,   3,   1]
        >>> val_list = ['a', 'b', 'c', 'd', 'e']
        >>> # execute function
        >>> key_to_vals = build_conflict_dict(key_list, val_list)
        >>> # verify results
        >>> result = ut.dict_str(key_to_vals)
        >>> print(result)
        {
            1: ['a', 'e'],
            2: ['b', 'c'],
            3: ['d'],
        }
    """
    key_to_vals = defaultdict(list)
    for key, val in zip(key_list, val_list):
        key_to_vals[key].append(val)
    return key_to_vals


def updateif_haskey(dict1, dict2):
    for key, val in six.iteritems(dict2):
        if key in dict1:
            dict1[key] = val


def dict_update_newkeys(dict_, dict2):
    """ Like dict.update, but does not overwrite items """
    for key, val in six.iteritems(dict2):
        if key not in dict_:
            dict_[key] = val


def is_dicteq(dict1_, dict2_, almosteq_ok=True, verbose_err=True):
    """ Checks to see if dicts are the same. Performs recursion. Handles numpy """
    import utool
    from utool import util_alg
    from utool import util_dbg
    assert len(dict1_) == len(dict2_), 'dicts are not of same length'
    try:
        for (key1, val1), (key2, val2) in zip(dict1_.items(), dict2_.items()):
            assert key1 == key2, 'key mismatch'
            assert type(val1) == type(val2), 'vals are not same type'
            if HAS_NUMPY and np.iterable(val1):
                if almosteq_ok and utool.is_float(val1):
                    assert np.all(util_alg.almost_eq(val1, val2)), 'float vals are not within thresh'
                else:
                    assert all([np.all(x1 == x2) for (x1, x2) in zip(val1, val2)]), 'np vals are different'
            elif isinstance(val1, dict):
                is_dicteq(val1, val2, almosteq_ok=almosteq_ok, verbose_err=verbose_err)
            else:
                assert val1 == val2, 'vals are different'
    except AssertionError as ex:
        if verbose_err:
            util_dbg.printex(ex)
        return False
    return True


def dict_subset(dict_, keys):
    r"""
    Args:
        dict_ (dict):
        keys (list):

    Returns:
        dict: subset dictionary

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = '?'
        >>> keys = '?'
        >>> result = dict_subset(dict_, keys)
        >>> print(result)
    """
    subdict_ = {key: dict_[key] for key in keys}
    return subdict_


def dict_setdiff(dict_, negative_keys):
    r"""
    Args:
        dict_ (dict):
        negative_keys (list):
    """
    keys = [key for key in six.iterkeys(dict_)
            if key not in set(negative_keys)]
    subdict_ = dict_subset(dict_, keys)
    return subdict_


def delete_dict_keys(dict_, key_list):
    invalid_keys = set(key_list) - set(six.iterkeys(dict_))
    for key in invalid_keys:
        del dict_[key]


def dict_take_gen(dict_, keys, *d):
    r"""
    Args:
        dict_ (dict):
        keys (list):

    Varargs:
        d: if specified is default for key errors

    CommandLine:
        python -m utool.util_dict --test-dict_take_gen

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {1: 'a', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> result = list(dict_take_gen(dict_, keys, None))
        >>> print(result)
        ['a', 'b', 'c', None, None]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {1: 'a', 2: 'b', 3: 'c'}
        >>> keys = [1, 2, 3, 4, 5]
        >>> try:
        >>>     print(list(dict_take_gen(dict_, keys)))
        >>>     result = 'did not get key error'
        >>> except KeyError:
        >>>     result = 'correctly got key error'
        >>> print(result)
        correctly got key error
    """
    if len(d) == 0:
        # no default given throws key error
        dictget = dict_.__getitem__
    elif len(d) == 1:
        # default given does not throw key erro
        dictget = dict_.get
    else:
        raise ValueError('len(d) must be 1 or 0')
    for key in keys:
        if HAS_NUMPY and isinstance(key, np.ndarray):
            # recursive call
            yield list(dict_take_gen(dict_, key, *d))
        else:
            yield dictget(key, *d)


def dict_take_list(dict_, keys, *d):
    return list(dict_take_gen(dict_, keys, *d))
    #return [dict_[key] for key in keys]


def dict_take(dict_, keys, *d):
    """ alias """
    return dict_take_list(dict_, keys, *d)


def dict_where_len0(dict_):
    keys = np.array(dict_.keys())
    flags = np.array(list(map(len, dict_.values()))) == 0
    indices = np.where(flags)[0]
    return keys[indices]


def dictinfo(dict_):
    """
    dictinfo

    In depth debugging info

    Args:
        dict_ (dict):

    Returns:
        str

    Example:
        >>> from utool.util_dict import *  # NOQA
        >>> dict_ = {}
        >>> result = dictinfo(dict_)
        >>> print(result)
    """
    import utool as ut
    if not isinstance(dict_, dict):
        return 'expected dict got %r' % type(dict_)

    keys = list(dict_.keys())
    vals = list(dict_.values())
    num_keys  = len(keys)
    key_types = list(set(map(type, keys)))
    val_types = list(set(map(type, vals)))

    fmtstr_ = '\n' + ut.unindent('''
    * num_keys  = {num_keys}
    * key_types = {key_types}
    * val_types = {val_types}
    '''.strip('\n'))

    if len(val_types) == 1:
        if val_types[0] == np.ndarray:
            # each key holds an ndarray
            val_shape_stats = ut.get_stats(set(map(np.shape, vals)), axis=0)
            val_shape_stats_str = ut.dict_str(val_shape_stats, strvals=True, newlines=False)
            val_dtypes = list(set([val.dtype for val in vals]))
            fmtstr_ += ut.unindent('''
            * val_shape_stats = {val_shape_stats_str}
            * val_dtypes = {val_dtypes}
            '''.strip('\n'))
        elif val_types[0] == list:
            # each key holds a list
            val_len_stats =  ut.get_stats(set(map(len, vals)))
            val_len_stats_str = ut.dict_str(val_len_stats, strvals=True, newlines=False)
            depth = ut.list_depth(vals)
            deep_val_types = list(set(ut.list_deep_types(vals)))
            fmtstr_ += ut.unindent('''
            * list_depth = {depth}
            * val_len_stats = {val_len_stats_str}
            * deep_types = {deep_val_types}
            '''.strip('\n'))
            if len(deep_val_types) == 1:
                if deep_val_types[0] == np.ndarray:
                    deep_val_dtypes = list(set([val.dtype for val in vals]))
                    fmtstr_ += ut.unindent('''
                    * deep_val_dtypes = {deep_val_dtypes}
                    ''').strip('\n')
        elif val_types[0] in [np.uint8, np.int8, np.int32, np.int64, np.float16, np.float32, np.float64]:
            # each key holds a scalar
            val_stats = ut.get_stats(vals)
            fmtstr_ += ut.unindent('''
            * val_stats = {val_stats}
            ''').strip('\n')

    fmtstr = fmtstr_.format(**locals())
    return ut.indent(fmtstr)


def dict_find_keys(dict_, val_list):
    r"""
    Args:
        dict_ (dict):
        val_list (list):

    Returns:
        dict: found_dict

    CommandLine:
        python -m utool.util_dict --test-dict_find_keys

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> # build test data
        >>> dict_ = {'default': 1, 'hierarchical': 5, 'linear': 0, 'kdtree': 1,
        ...          'composite': 3, 'autotuned': 255, 'saved': 254, 'kmeans': 2,
        ...          'lsh': 6, 'kdtree_single': 4}
        >>> val_list = [1]
        >>> # execute function
        >>> found_dict = dict_find_keys(dict_, val_list)
        >>> # verify results
        >>> result = str(found_dict)
        >>> print(result)
        {1: ['kdtree', 'default']}
    """
    found_dict = {
        search_val: [key for key, val in six.iteritems(dict_)
                     if val == search_val]
        for search_val in val_list
    }
    return found_dict


def dict_find_other_sameval_keys(dict_, key):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_dict import *  # NOQA
        >>> # build test data
        >>> dict_ = {'default': 1, 'hierarchical': 5, 'linear': 0, 'kdtree': 1,
        ...          'composite': 3, 'autotuned': 255, 'saved': 254, 'kmeans': 2,
        ...          'lsh': 6, 'kdtree_single': 4}
        >>> key = 'default'
        >>> # execute function
        >>> found_dict = dict_find_keys(dict_, val_list)
    """
    value = dict_[key]
    found_dict = dict_find_keys(dict_, [value])
    other_keys = found_dict[value]
    other_keys.remove(key)
    return other_keys


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_dict; utool.doctest_funcs(utool.util_dict, allexamples=True)"
        python -c "import utool, utool.util_dict; utool.doctest_funcs(utool.util_dict)"
        python -m utool.util_dict
        python -m utool.util_dict --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
