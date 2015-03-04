""" module for gridsearch helper """
from __future__ import absolute_import, division, print_function
from collections import namedtuple, OrderedDict
from utool import util_class
from utool import util_inject
from utool import util_dict
from utool import util_dev
import six
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[gridsearch]')


DimensionBasis = namedtuple('DimensionBasis', ('dimension_name', 'dimension_point_list'))


@six.add_metaclass(util_class.ReloadingMetaclass)
class ParamInfo(object):
    """ small class for individual paramater information """
    def __init__(pi, varname, default, shortprefix=util_dev.NoParam,
                 type_=util_dev.NoParam, varyvals=[], varyslice=None,
                 hideif=util_dev.NoParam):
        r"""
        Args:
            varname (?):
            default (?):
            shortprefix (ClassNoParam):
            type_ (ClassNoParam):
            varyvals (list):
            hideif (var): if the variable value of config is this the itemstr is empty

        CommandLine:
            python -m utool.util_gridsearch --test-__init__

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_gridsearch import *  # NOQA
            >>> # build test data
            >>> pi = '?'
            >>> varname = '?'
            >>> default = '?'
            >>> shortprefix = <utool.util_dev.ClassNoParam object at 0x7f2826164490>
            >>> type_ = <utool.util_dev.ClassNoParam object at 0x7f2826164490>
            >>> varyvals = []
            >>> # execute function
            >>> result = pi.__init__(varname, default, shortprefix, type_, varyvals)
            >>> # verify results
            >>> print(result)
        """
        pi.varname = varname
        pi.default = default
        pi.shortprefix = shortprefix
        pi.type_ = type(default) if type_ is util_dev.NoParam else type_
        # for gridsearch
        pi.varyvals = varyvals
        pi.varyslice = varyslice
        pi.hideif_list = []
        if hideif is not util_dev.NoParam:
            pi.append_hideif(hideif)

    def append_hideif(pi, hideif):
        pi.hideif_list.append(hideif)

    def is_hidden(pi, cfg):
        for hideif in pi.hideif_list:
            if hasattr(hideif, '__call__'):
                hide = hideif(cfg)
            else:
                hide = getattr(cfg,  pi.varname) == hideif
            if hide:
                return True

    def make_itemstr(pi, cfg):
        varval = getattr(cfg,  pi.varname)
        varstr = str(varval)
        if pi.shortprefix is not util_dev.NoParam:
            itemstr = pi.shortprefix + varstr
        else:
            itemstr =  pi.varname + '=' + varstr
        return itemstr

    def get_itemstr(pi, cfg):
        if pi.is_hidden(cfg):
            return ''
        else:
            return pi.make_itemstr(cfg)


@six.add_metaclass(util_class.ReloadingMetaclass)
class ParamInfoBool(ParamInfo):
    def __init__(pi, varname, default=False, shortprefix=util_dev.NoParam, type_=bool, varyvals=[], varyslice=None, hideif=False):
        super(ParamInfoBool, pi).__init__(
            varname, default=default, shortprefix=shortprefix, type_=bool,
            varyvals=varyvals, varyslice=varyslice, hideif=hideif)

    def make_itemstr(pi, cfg):
        varval = getattr(cfg,  pi.varname)
        if pi.shortprefix is not util_dev.NoParam:
            itemstr = pi.shortprefix
        else:
            itemstr =  pi.varname.replace('_on', '')
        if varval is False:
            itemstr = 'no' + itemstr
        elif varval is not True:
            raise AssertionError('Not a boolean pi.varname=%r, varval=%r' % (pi.varname, varval,))
        return itemstr


@six.add_metaclass(util_class.ReloadingMetaclass)
class ParamInfoList(object):
    """ small class for ut.Pref-less configurations """
    def __init__(self, name, param_info_list=[], constraint_func=None, hideif=None):
        self.name = name
        self.param_info_list = param_info_list
        self.constraint_func = constraint_func
        if hideif is not None:
            self.append_hideif(hideif)

    def append_hideif(self, hideif):
        # apply hideif to all children
        for pi in self.param_info_list:
            pi.append_hideif(hideif)

    def aslist(self, hideif=None):
        if hideif is not None:
            self.append_hideif(hideif)
        return self.param_info_list

    def updated_cfgdict(self, dict_):
        return {pi.varname: dict_.get(pi.varname, pi.default) for pi in self.param_info_list}

    def get_varnames(self):
        return [pi.varname for pi in self.param_info_list]

    def get_varydict(self):
        """ for gridsearch """
        varied_dict = {pi.varname: pi.varyvals for pi in self.param_info_list}
        return varied_dict

    def get_slicedict(self):
        """ for gridsearch """
        slice_dict = {pi.varname: pi.varyslice for pi in self.param_info_list if pi.varyslice is not None}
        if len(slice_dict) == 0:
            slice_dict = None
        return slice_dict

    def get_grid_basis(self):
        """ DEPRICATE """
        grid_basis = [
            DimensionBasis(pi.varname, pi.varyvals)
            for pi in self.param_info_list
        ]
        return grid_basis

    def get_gridsearch_input(self, defaultslice=slice(0, 1)):
        """ for gridsearch """
        varied_dict = self.get_varydict()
        slice_dict = self.get_slicedict()
        constraint_func = self.constraint_func
        cfgdict_list, cfglbl_list = make_constrained_cfg_and_lbl_list(
            varied_dict,
            constraint_func=constraint_func,
            slice_dict=slice_dict,
            defaultslice=defaultslice
        )
        return cfgdict_list, cfglbl_list


def testdata_grid_search():
    """
    test data function for doctests
    """
    import utool as ut
    grid_basis = [
        ut.DimensionBasis('p', [.5, .8, .9, 1.0]),
        ut.DimensionBasis('K', [2, 3, 4, 5]),
        ut.DimensionBasis('dcvs_clip_max', [.1, .2, .5, 1.0]),
    ]
    gridsearch = ut.GridSearch(grid_basis, label='testdata_gridsearch')
    for cfgdict in gridsearch:
        tp_score = cfgdict['p'] + (cfgdict['K'] ** .5)
        tn_score = (cfgdict['p'] * (cfgdict['K'])) / cfgdict['dcvs_clip_max']
        gridsearch.append_result(tp_score, tn_score)
    return gridsearch


@six.add_metaclass(util_class.ReloadingMetaclass)
class GridSearch(object):
    """
    helper for executing iterations and analyzing the results of a grid search

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> grid_basis = [
        ...     ut.DimensionBasis('p', [.5, .8, .9, 1.0]),
        ...     ut.DimensionBasis('K', [2, 3, 4, 5]),
        ...     ut.DimensionBasis('dcvs_clip_max', [.1, .2, .5, 1.0]),
        ... ]
        >>> gridsearch = ut.GridSearch(grid_basis, label='testdata_gridsearch')
        >>> for cfgdict in gridsearch:
        ...    tp_score = cfgdict['p'] + (cfgdict['K'] ** .5)
        ...    tn_score = (cfgdict['p'] * (cfgdict['K'])) / cfgdict['dcvs_clip_max']
        ...    gridsearch.append_result(tp_score, tn_score)
    """
    def __init__(gridsearch, grid_basis, label=None):
        gridsearch.label = label
        gridsearch.grid_basis = grid_basis
        gridsearch.tp_score_list = []
        gridsearch.tn_score_list = []
        gridsearch.score_diff_list = []
        cfgdict_iter = grid_search_generator(grid_basis)
        gridsearch.cfgdict_list = list(cfgdict_iter)
        gridsearch.num_configs = len(gridsearch.cfgdict_list)
        gridsearch.score_lbls  = ['score_diff', 'tp_score', 'tn_score']

    def append_result(gridsearch, tp_score, tn_score):
        """ for use in iteration """
        diff = tp_score - tn_score
        gridsearch.score_diff_list.append(diff)
        gridsearch.tp_score_list.append(tp_score)
        gridsearch.tn_score_list.append(tn_score)

    def __iter__(gridsearch):
        for cfgdict in gridsearch.cfgdict_list:
            yield cfgdict

    def __len__(gridsearch):
        return gridsearch.num_configs

    def get_score_list_and_lbls(gridsearch):
        """ returns result data """
        score_list  = [gridsearch.score_diff_list,
                       gridsearch.tp_score_list,
                       gridsearch.tn_score_list]
        score_lbls = gridsearch.score_lbls
        return score_list, score_lbls

    def get_param_list_and_lbls(gridsearch):
        """ returns input data """
        import utool as ut
        param_name_list = ut.get_list_column(gridsearch.grid_basis, 0)
        params_vals = [list(six.itervalues(dict_)) for dict_ in gridsearch.cfgdict_list]
        param_vals_list = list(zip(*params_vals))
        return param_name_list, param_vals_list

    def get_param_lbls(gridsearch, exclude_unvaried_dimension=True):
        #param_label_list, param_vals_list = gridsearch.get_param_list_and_lbls()[0]
        import utool as ut
        param_label_list_ = ut.get_list_column(gridsearch.grid_basis, 0)
        param_range_list = ut.get_list_column(gridsearch.grid_basis, 1)
        if exclude_unvaried_dimension:
            is_varied = [len(param_range) > 1 for param_range in param_range_list]
            param_label_list = ut.filter_items(param_label_list_, is_varied)
        else:
            param_label_list = param_label_list_
        return param_label_list

    def get_sorted_columns_and_labels(gridsearch, score_lbl='score_diff'):
        """ returns sorted input and result data """
        import utool as ut
        # Input Parameters
        param_name_list, param_vals_list = gridsearch.get_param_list_and_lbls()
        # Result Scores
        score_list, score_lbls = gridsearch.get_score_list_and_lbls()

        score_vals = score_list[score_lbls.index(score_lbl)]
        sortby_func = ut.make_sortby_func(score_vals, reverse=True)

        score_name_sorted = score_lbls
        param_name_sorted = param_name_list
        score_list_sorted = list(map(sortby_func, score_list))
        param_vals_sorted = list(map(sortby_func, param_vals_list))
        collbl_tup = (score_name_sorted, param_name_sorted,
                      score_list_sorted, param_vals_sorted)
        return collbl_tup

    def get_csv_results(gridsearch, max_lines=None, score_lbl='score_diff'):
        """
        Make csv text describing results

        Args:
            max_lines (int): add top num lines to the csv. No limit if None.
            score_lbl (str): score label to sort by

        Returns:
            str: result data in csv format

        CommandLine:
            python -m utool.util_gridsearch --test-get_csv_results

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_gridsearch import *  # NOQA
            >>> import utool as ut
            >>> import plottool as pt
            >>> # build test data
            >>> score_lbl = 'score_diff'
            >>> gridsearch = testdata_grid_search()
            >>> csvtext = gridsearch.get_csv_results(10, score_lbl)
            >>> print(csvtext)
            >>> result = ut.hashstr(csvtext)
            >>> print(result)
            60yptleiwo@lk@24
        """
        import utool as ut
        collbl_tup = gridsearch.get_sorted_columns_and_labels(score_lbl)
        (score_name_sorted, param_name_sorted,
         score_list_sorted, param_vals_sorted) = collbl_tup

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
            label = {label}
            grid_basis = {grid_basis_str}
            ''')
        fmtdict = dict(
            grid_basis_str=ut.list_str(gridsearch.grid_basis),
            label=gridsearch.label
        )
        header_raw = header_raw_fmtstr.format(**fmtdict)
        header     = ut.indent(header_raw, '# >>> ')
        precision = 3
        csvtext = ut.make_csv_table(column_list, column_lbls, header, precision=precision)
        return csvtext

    def get_rank_cfgdict(gridsearch, rank=0, score_lbl='score_diff'):
        import utool as ut
        collbl_tup = gridsearch.get_sorted_columns_and_labels(score_lbl)
        (score_name_sorted, param_name_sorted,
         score_list_sorted, param_vals_sorted) = collbl_tup
        rank_vals = ut.get_list_column(param_vals_sorted, rank)
        rank_cfgdict = dict(zip(param_name_sorted, rank_vals))
        return rank_cfgdict

    def get_dimension_stats(gridsearch, param_lbl, score_lbl='score_diff'):
        r"""
        Returns result stats about a specific parameter

        Args:
            param_lbl (str); paramter to get stats about
            score_lbl (str): score label to sort by

        Returns:
            dict: param2_score_stats
        """
        import utool as ut
        score_list, score_lbls = gridsearch.get_score_list_and_lbls()
        param_name_list, param_vals_list = gridsearch.get_param_list_and_lbls()
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

    def get_dimension_stats_str(gridsearch, param_lbl, score_lbl='score_diff'):
        r"""
        Returns a result stat string about a specific parameter
        """
        import utool as ut
        exclude_keys = ['nMin', 'nMax']
        param2_score_stats = gridsearch.get_dimension_stats(param_lbl)
        param2_score_stats_str = {
            param: ut.get_stats_str(stat_dict=stat_dict, exclude_keys=exclude_keys)
            for param, stat_dict in six.iteritems(param2_score_stats)}
        param_stats_str = 'stats(' + param_lbl + ') = ' + ut.dict_str(param2_score_stats_str)
        return param_stats_str

    def plot_dimension(gridsearch, param_lbl, score_lbl='score_diff',
                       **kwargs):
        r"""
        Plots result statistics about a specific parameter

        Args:
            param_lbl (str);
            score_lbl (str):

        CommandLine:
            python -m utool.util_gridsearch --test-plot_dimension
            python -m utool.util_gridsearch --test-plot_dimension --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from utool.util_gridsearch import *  # NOQA
            >>> import plottool as pt
            >>> # build test data
            >>> gridsearch = testdata_grid_search()
            >>> param_lbl = 'p'
            >>> score_lbl = 'score_diff'
            >>> self = gridsearch
            >>> self.plot_dimension('p', score_lbl, fnum=1, pnum=(1, 3, 1))
            >>> self.plot_dimension('K', score_lbl, fnum=1, pnum=(1, 3, 2))
            >>> self.plot_dimension('dcvs_clip_max', score_lbl, fnum=1, pnum=(1, 3, 3))
            >>> pt.show_if_requested()
        """
        import plottool as pt
        param2_score_stats = gridsearch.get_dimension_stats(param_lbl, score_lbl)
        title = param_lbl
        #title = param_lbl + ' vs ' + score_lbl
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


def make_constrained_cfg_and_lbl_list(varied_dict, constraint_func=None,
                                      slice_dict=None,
                                      defaultslice=slice(0, 1)):
    r"""
    Args:
        varied_dict (dict):  parameters to vary with possible variations
        constraint_func (func): function to restirct parameter variations
        slice_dict (dict):  dict of slices for each param of valid possible values
        defaultslice (slice): default slice used if slice is not specified in slice_dict

    Returns:
        tuple: (cfgdict_list, cfglbl_list)

    CommandLine:
        python -m utool.util_gridsearch --test-make_constrained_cfg_and_lbl_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> # build test data
        >>> varied_dict = {
        ...     'p': [.1, .3, 1.0, 2.0],
        ...     'dcvs_clip_max': [.1, .2, .5],
        ...     'K': [3, 5],
        ... }
        >>> constraint_func = None
        >>> # execute function
        >>> (cfgdict_list, cfglbl_list) = make_constrained_cfg_and_lbl_list(varied_dict, constraint_func)
        >>> # verify results
        >>> result = str((cfgdict_list, cfglbl_list))
        >>> print(result)
    """
    # Restrict via slices
    if slice_dict is None:
        varied_dict_ = varied_dict
    else:
        varied_dict_ = {
            key: val[slice_dict.get(key, defaultslice)]
            for key, val in six.iteritems(varied_dict)
        }
    # Enumerate all combinations
    cfgdict_list_ = util_dict.all_dict_combinations(varied_dict_)
    if constraint_func is not None:
        # Remove invalid combinations
        cfgdict_list = constrain_cfgdict_list(cfgdict_list_, constraint_func)
    else:
        cfgdict_list = cfgdict_list_
    # Get labels and return
    cfglbl_list = make_cfglbls(cfgdict_list, varied_dict)
    return cfgdict_list, cfglbl_list


def get_cfgdict_lbl_list_subset(cfgdict_list, varied_dict):
    keys = list(varied_dict.iterkeys())
    cfgdict_sublist = get_cfgdict_list_subset(cfgdict_list, keys)
    cfglbl_sublist = make_cfglbls(cfgdict_sublist, varied_dict)
    return cfgdict_sublist, cfglbl_sublist


def get_cfgdict_list_subset(cfgdict_list, keys):
    r"""
    returns list of unique dictionaries only with keys specified in keys

    Args:
        cfgdict_list (list):
        keys (list):

    Returns:
        list: cfglbl_list

    CommandLine:
        python -m utool.util_gridsearch --test-get_cfgdict_list_subset

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> cfgdict_list = [
        ...    {'K': 3, 'dcvs_clip_max': 0.1, 'p': 0.1},
        ...    {'K': 5, 'dcvs_clip_max': 0.1, 'p': 0.1},
        ...    {'K': 5, 'dcvs_clip_max': 0.1, 'p': 0.2},
        ...    {'K': 3, 'dcvs_clip_max': 0.2, 'p': 0.1},
        ...    {'K': 5, 'dcvs_clip_max': 0.2, 'p': 0.1},
        ...    {'K': 3, 'dcvs_clip_max': 0.2, 'p': 0.1}]
        >>> keys = ['K', 'dcvs_clip_max']
        >>> # execute function
        >>> cfgdict_sublist = get_cfgdict_list_subset(cfgdict_list, keys)
        >>> # verify results
        >>> result = ut.list_str(cfgdict_sublist)
        >>> print(result)
        [
            {'K': 3, 'dcvs_clip_max': 0.1,},
            {'K': 5, 'dcvs_clip_max': 0.1,},
            {'K': 3, 'dcvs_clip_max': 0.2,},
            {'K': 5, 'dcvs_clip_max': 0.2,},
        ]
    """
    import utool as ut
    cfgdict_sublist_ = [ut.dict_subset(cfgdict, keys) for cfgdict in cfgdict_list]
    cfgtups_sublist_ = [tuple(ut.dict_to_keyvals(cfgdict)) for cfgdict in cfgdict_sublist_]
    cfgtups_sublist = ut.unique_keep_order2(cfgtups_sublist_)
    cfgdict_sublist = list(map(dict, cfgtups_sublist))
    return cfgdict_sublist


def constrain_cfgdict_list(cfgdict_list_, constraint_func):
    """ constrains configurations and removes duplicates """
    cfgdict_list = []
    for cfg_ in cfgdict_list_:
        cfg = cfg_.copy()
        if constraint_func(cfg) is not False and len(cfg) > 0:
            if cfg not in cfgdict_list:
                cfgdict_list.append(cfg)
    return cfgdict_list


def make_cfglbls(cfgdict_list, varied_dict):
    """  Show only the text in labels that mater from the cfgdict """
    import textwrap
    wrapper = textwrap.TextWrapper(width=50)
    cfglbl_list =  []
    for cfgdict_ in cfgdict_list:
        cfgdict = cfgdict_.copy()
        for key in six.iterkeys(cfgdict_):
            try:
                vals = varied_dict[key]
                # Dont print label if not varied
                if len(vals) == 1:
                    del cfgdict[key]
                else:
                    # Dont print label if it is None (irrelevant)
                    if cfgdict[key] is None:
                        del cfgdict[key]
            except KeyError:
                # Don't print keys not in varydict
                del cfgdict[key]
        cfglbl = str(cfgdict)
        search_repl_list = [('\'', ''), ('}', ''),
                            ('{', ''), (': ', '=')]
        for search, repl in search_repl_list:
            cfglbl = cfglbl.replace(search, repl)
        #cfglbl = str(cfgdict).replace('\'', '').replace('}', '').replace('{', '').replace(': ', '=')
        cfglbl = ('\n'.join(wrapper.wrap(cfglbl)))
        cfglbl_list.append(cfglbl)
    return cfglbl_list


def interact_gridsearch_result_images(show_result_func, cfgdict_list,
                                      cfglbl_list, cfgresult_list,
                                      score_list=None, fnum=None, figtitle='',
                                      unpack=False, max_plots=25, verbose=True,
                                      precision=3, scorelbl='score',
                                      onclick_func=None):
    """ helper function for visualizing results of gridsearch """
    assert hasattr(show_result_func, '__call__'), 'NEED FUNCTION GOT: %r' % (show_result_func,)

    import utool as ut
    import plottool as pt
    from plottool import plot_helpers as ph
    from plottool import interact_helpers as ih
    if verbose:
        print('Plotting gridsearch results figtitle=%r' % (figtitle,))
    if score_list is None:
        score_list = [None] * len(cfgdict_list)
    else:
        # sort by score if available
        sortx_list = ut.list_argsort(score_list, reverse=True)
        score_list = ut.list_take(score_list, sortx_list)
        cfgdict_list = ut.list_take(cfgdict_list, sortx_list)
        cfglbl_list = ut.list_take(cfglbl_list, sortx_list)
        cfgresult_list = ut.list_take(cfgresult_list, sortx_list)
    # Dont show too many results only the top few
    score_list = ut.listclip(score_list, max_plots)

    # Show the config results
    fig = pt.figure(fnum=fnum)
    # Get plots for each of the resutls
    nRows, nCols = pt.get_square_row_cols(len(score_list), fix=True)
    next_pnum = pt.make_pnum_nextgen(nRows, nCols)
    for cfgdict, cfglbl, cfgresult, score in zip(cfgdict_list, cfglbl_list,
                                                  cfgresult_list,
                                                  score_list):
        if score is not None:
            cfglbl += '\n' + scorelbl + '=' + ut.numeric_str(score, precision=precision)
        pnum = next_pnum()
        try:
            if unpack:
                show_result_func(*cfgresult, fnum=fnum, pnum=pnum)
            else:
                show_result_func(cfgresult, fnum=fnum, pnum=pnum)
        except Exception as ex:
            if isinstance(cfgresult, tuple):
                #print(ut.list_str(cfgresult))
                print(ut.depth_profile(cfgresult))
                print(ut.list_type_profile(cfgresult))
            ut.printex(ex, 'error showing', keys=['cfgresult', 'fnum', 'pnum'])
            raise
        #pt.imshow(255 * cfgresult, fnum=fnum, pnum=next_pnum(), title=cfglbl)
        ax = pt.gca()
        pt.set_title(cfglbl, ax=ax)  # , size)
        ph.set_plotdat(ax, 'cfgdict', cfgdict)
        ph.set_plotdat(ax, 'cfglbl', cfglbl)
        ph.set_plotdat(ax, 'cfgresult', cfgresult)
    # Define clicked callback
    def on_clicked(event):
        print('\n[pt] clicked gridsearch axes')
        if event is None or event.xdata is None or event.inaxes is None:
            print('out of axes')
            pass
        else:
            ax = event.inaxes
            plotdat_dict = ph.get_plotdat_dict(ax)
            print(ut.dict_str(plotdat_dict))
            cfglbl = ph.get_plotdat(ax, 'cfglbl', None)
            cfgdict = ph.get_plotdat(ax, 'cfgdict', {})
            cfgresult = ph.get_plotdat(ax, 'cfgresult', {})
            infostr_list = [
                ('cfglbl = ' + str(cfglbl)),
                '',
                ('cfgdict = ' + ut.dict_str(cfgdict, sorted_=True)),
            ]
            # Call a user defined function if given
            if onclick_func is not None:
                if unpack:
                    onclick_func(*cfgresult)
                else:
                    onclick_func(cfgresult)
            infostr = ut.msgblock('CLICKED', '\n'.join(infostr_list))
            print(infostr)
    # Connect callbacks
    ih.connect_callback(fig, 'button_press_event', on_clicked)
    pt.set_figtitle(figtitle)


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
