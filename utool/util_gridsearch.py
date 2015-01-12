""" module for gridsearch helper """
from __future__ import absolute_import, division, print_function
from collections import namedtuple, OrderedDict
from utool import util_class
from utool import util_inject
from utool import util_dict
import six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[gridsearch]')


DimensionBasis = namedtuple('DimensionBasis', ('dimension_name', 'dimension_point_list'))


def testdata_grid_search():
    import utool as ut
    grid_basis = [
        ut.util_gridsearch.DimensionBasis('p', [.5, .6, .7, .8, .9, 1.0]),
        ut.util_gridsearch.DimensionBasis('K', [2, 3, 4, 5]),
        ut.util_gridsearch.DimensionBasis('clip_fraction', [.1, .2, .5, 1.0]),
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
            python -m utool.util_gridsearch --test-plot_dimension
            python -m utool.util_gridsearch --test-plot_dimension --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_gridsearch import *  # NOQA
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


def grid_search_generator(grid_basis=[], *args, **kwargs):
    r"""
    Iteratively yeilds individual configuration points
    inside a defined basis.

    Args:
        grid_basis (list): a list of 2-component tuple. The named tuple looks
            like this:

    CommandLine:
        python -m utool.util_gridsearch --test-grid_search_generator

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
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
        # num_rows=3
        #   dim1,  dim2
            0.10,  0.10
            0.20,  0.40
            0.30,  0.50

    """
    grid_basis_ = grid_basis + list(args) + list(kwargs.items())
    grid_basis_dict = OrderedDict(grid_basis_)
    grid_point_iter = util_dict.iter_all_dict_combinations_ordered(grid_basis_dict)
    for grid_point in grid_point_iter:
        yield grid_point


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_gridsearch
        python -m utool.util_gridsearch --allexamples
        python -m utool.util_gridsearch --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
