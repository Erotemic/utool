# -*- coding: utf-8 -*-
""" module for gridsearch helper """
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple, OrderedDict
from utool import util_class
from utool import util_inject
from utool import util_dict
from utool import util_dev
from utool import util_const
from utool import util_decor
from utool import util_type
import functools
import operator
from six.moves import reduce, map, zip
import re
import six
print, rrr, profile = util_inject.inject2(__name__)


DimensionBasis = namedtuple('DimensionBasis', ('dimension_name', 'dimension_point_list'))

INTERNAL_CFGKEYS = ['_cfgstr', '_cfgname', '_cfgtype', '_cfgindex']
NAMEVARSEP = ':'


@six.add_metaclass(util_class.ReloadingMetaclass)
class CountstrParser(object):
    """
    Parses a statement like  '#primary>0&#primary1>1' and returns a filtered
    set.

    FIXME: make generalizable beyond ibeis
    """
    numop = '#'
    compare_op_map = {
        '<'  : operator.lt,
        '<=' : operator.le,
        '>'  : operator.gt,
        '>=' : operator.ge,
        '='  : operator.eq,
        '!=' : operator.ne,
    }

    def __init__(self, lhs_dict, prop2_nid2_aids):
        self.lhs_dict = lhs_dict
        self.prop2_nid2_aids = prop2_nid2_aids
        pass

    def parse_countstr_binop(self, part):
        import utool as ut
        import numpy as np
        # Parse binary comparison operation
        left, op, right = re.split(ut.regex_or(('[<>]=?', '=')), part)
        # Parse length operation. Get prop_left_nids, prop_left_values
        if left.startswith(self.numop):
            varname = left[len(self.numop):]
            # Parse varname
            prop = self.lhs_dict.get(varname, varname)
            # Apply length operator to each name with the prop
            prop_left_nids = self.prop2_nid2_aids.get(prop, {}).keys()
            valiter = self.prop2_nid2_aids.get(prop, {}).values()
            prop_left_values = np.array(list(map(len, valiter)))
        # Pares number
        if right:
            prop_right_value = int(right)
        # Execute comparison
        prop_binary_result = self.compare_op_map[op](
            prop_left_values, prop_right_value)
        prop_nid2_result = dict(zip(prop_left_nids, prop_binary_result))
        return prop_nid2_result

    def parse_countstr_expr(self, countstr):
        # Split over ands for now
        and_parts = countstr.split('&')
        prop_nid2_result_list = []
        for part in and_parts:
            prop_nid2_result = self.parse_countstr_binop(part)
            prop_nid2_result_list.append(prop_nid2_result)
        import utool as ut
        # change to dict_union when parsing ors
        andcombine = functools.partial(
            ut.dict_isect_combine, combine_op=operator.and_)
        expr_nid2_result = reduce(andcombine, prop_nid2_result_list)
        return expr_nid2_result


def parse_argv_cfg(argname, default=[''], named_defaults_dict=None,
                   valid_keys=None, alias_keys=None):
    """
    simple configs

    Args:
        argname (str):
        default (list): (default = [])
        named_defaults_dict (dict): (default = None)
        valid_keys (None): (default = None)

    Returns:
        list: cfg_list

    CommandLine:
        python -m utool.util_gridsearch --exec-parse_argv_cfg --filt :foo=bar
        python -m utool.util_gridsearch --exec-parse_argv_cfg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> argname = '--filt'
        >>> cfg_list = parse_argv_cfg(argname)
        >>> result = ('cfg_list = %s' % (six.text_type(cfg_list),))
        >>> print(result)
    """
    import utool as ut
    if ut.in_jupyter_notebook():
        # dont parse argv in ipython notebook
        cfgstr_list = default
    else:
        cfgstr_list = ut.get_argval(argname, type_=list, default=default)
    if cfgstr_list is None:
        return None
    cfg_combos_list = parse_cfgstr_list2(cfgstr_list,
                                         named_defaults_dict=named_defaults_dict,
                                         valid_keys=valid_keys,
                                         alias_keys=alias_keys,
                                         strict=False)
    cfg_list = ut.flatten(cfg_combos_list)
    return cfg_list


def get_varied_cfg_lbls(cfg_list, default_cfg=None, mainkey='_cfgname',
                        checkname=False):
    r"""
    Args:
        cfg_list (list):
        default_cfg (None): (default = None)
        checkname (bool): if True removes names if they are all the same.

    Returns:
        list: cfglbl_list

    CommandLine:
        python -m utool.util_gridsearch --exec-get_varied_cfg_lbls

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfg_list = [{'_cfgname': 'test', 'f': 1, 'b': 1},
        >>>             {'_cfgname': 'test', 'f': 2, 'b': 1},
        >>>             {'_cfgname': 'test', 'f': 3, 'b': 1, 'z': 4}]
        >>> default_cfg = None
        >>> cfglbl_list = get_varied_cfg_lbls(cfg_list, default_cfg)
        >>> result = ('cfglbl_list = %s' % (ut.repr2(cfglbl_list),))
        >>> print(result)
        cfglbl_list = ['test:f=1', 'test:f=2', 'test:f=3,z=4']
    """
    import utool as ut
    try:
        cfgname_list = [cfg[mainkey] for cfg in cfg_list]
    except KeyError:
        cfgname_list = [''] * len(cfg_list)
    varied_cfg_list = partition_varied_cfg_list(cfg_list, default_cfg)[1]
    if checkname and ut.allsame(cfgname_list):
        cfgname_list = [None] * len(cfgname_list)
    cfglbl_list = [get_cfg_lbl(cfg, name)
                   for cfg, name in zip(varied_cfg_list, cfgname_list)]
    if checkname:
        cfglbl_list = [x.lstrip(':') for x in cfglbl_list]
    return cfglbl_list


def get_nonvaried_cfg_lbls(cfg_list, default_cfg=None, mainkey='_cfgname'):
    r"""
    TODO: this might only need to return a single value. Maybe not if the names
        are different.

    Args:
        cfg_list (list):
        default_cfg (None): (default = None)

    Returns:
        list: cfglbl_list

    CommandLine:
        python -m utool.util_gridsearch --exec-get_nonvaried_cfg_lbls

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfg_list = [{'_cfgname': 'test', 'f': 1, 'b': 1},
        >>>             {'_cfgname': 'test', 'f': 2, 'b': 1},
        >>>             {'_cfgname': 'test', 'f': 3, 'b': 1, 'z': 4}]
        >>> default_cfg = None
        >>> cfglbl_list = get_nonvaried_cfg_lbls(cfg_list, default_cfg)
        >>> result = ('cfglbl_list = %s' % (ut.repr2(cfglbl_list),))
        >>> print(result)
        cfglbl_list = ['test:b=1', 'test:b=1', 'test:b=1']
    """
    try:
        cfgname_list = [cfg[mainkey] for cfg in cfg_list]
    except KeyError:
        cfgname_list = [''] * len(cfg_list)
    nonvaried_cfg = partition_varied_cfg_list(cfg_list, default_cfg)[0]
    cfglbl_list = [get_cfg_lbl(nonvaried_cfg, name) for name in cfgname_list]
    return cfglbl_list


def partition_varied_cfg_list(cfg_list, default_cfg=None, recursive=False):
    r"""
    Separates varied from non-varied parameters in a list of configs

    TODO: partition nested configs

    CommandLine:
        python -m utool.util_gridsearch --exec-partition_varied_cfg_list:0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfg_list = [{'f': 1, 'b': 1}, {'f': 2, 'b': 1}, {'f': 3, 'b': 1, 'z': 4}]
        >>> nonvaried_cfg, varied_cfg_list = partition_varied_cfg_list(cfg_list)
        >>> result = ut.repr4({'nonvaried_cfg': nonvaried_cfg,
        >>>                    'varied_cfg_list': varied_cfg_list}, explicit=1, nobr=True, nl=1)
        >>> print(result)
        nonvaried_cfg={'b': 1},
        varied_cfg_list=[{'f': 1}, {'f': 2}, {'f': 3, 'z': 4}],

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfg_list = [{'q1': 1, 'f1': {'a2': {'x3': 1, 'y3': 2}, 'b2': 1}}, {'q1': 1, 'f1': {'a2': {'x3': 1, 'y3': 1}, 'b2': 1}, 'e1': 1}]
        >>> print(ut.repr4(cfg_list, nl=1))
        >>> nonvaried_cfg, varied_cfg_list = partition_varied_cfg_list(cfg_list, recursive=True)
        >>> result = ut.repr4({'nonvaried_cfg': nonvaried_cfg,
        >>>                    'varied_cfg_list': varied_cfg_list}, explicit=1, nobr=True, nl=1)
        >>> print(result)
        nonvaried_cfg={'f1': {'a2': {'x3': 1}, 'b2': 1}, 'q1': 1},
        varied_cfg_list=[{'f1': {'a2': {'y3': 2}}}, {'e1': 1, 'f1': {'a2': {'y3': 1}}}],
    """
    import utool as ut
    if default_cfg is None:
        nonvaried_cfg = reduce(ut.dict_intersection, cfg_list)
    else:
        nonvaried_cfg = reduce(ut.dict_intersection, [default_cfg] + cfg_list)
    nonvaried_keys = list(nonvaried_cfg.keys())
    varied_cfg_list = [
        ut.delete_dict_keys(cfg.copy(), nonvaried_keys)
        for cfg in cfg_list]
    if recursive:
        # Find which varied keys have dict values
        varied_keys = list(set([key for cfg in varied_cfg_list for key in cfg]))
        varied_vals_list = [[cfg[key] for cfg in varied_cfg_list if key in cfg] for key in varied_keys]
        for key, varied_vals in zip(varied_keys, varied_vals_list):
            if len(varied_vals) == len(cfg_list):
                if all([isinstance(val, dict) for val in varied_vals]):
                    nonvaried_subdict, varied_subdicts = partition_varied_cfg_list(varied_vals, recursive=recursive)
                    nonvaried_cfg[key] = nonvaried_subdict
                    for cfg, subdict in zip(varied_cfg_list, varied_subdicts):
                        cfg[key] = subdict
    return nonvaried_cfg, varied_cfg_list


def get_cfg_lbl(cfg, name=None, nonlbl_keys=INTERNAL_CFGKEYS, key_order=None,
                with_name=True, default_cfg=None, sep=''):
    r"""
    Formats a flat configuration dict into a short string label. This is useful
    for re-creating command line strings.

    Args:
        cfg (dict):
        name (str): (default = None)
        nonlbl_keys (list): (default = INTERNAL_CFGKEYS)

    Returns:
        str: cfg_lbl

    CommandLine:
        python -m utool.util_gridsearch get_cfg_lbl

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfg = {'_cfgname': 'test', 'var1': 'val1', 'var2': 'val2'}
        >>> name = None
        >>> nonlbl_keys = ['_cfgstr', '_cfgname', '_cfgtype', '_cfgindex']
        >>> cfg_lbl = get_cfg_lbl(cfg, name, nonlbl_keys)
        >>> result = ('cfg_lbl = %s' % (six.text_type(cfg_lbl),))
        >>> print(result)
        cfg_lbl = test:var1=val1,var2=val2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfg = {'var1': 'val1', 'var2': 'val2'}
        >>> default_cfg = {'var2': 'val1', 'var1': 'val1'}
        >>> name = None
        >>> cfg_lbl = get_cfg_lbl(cfg, name, default_cfg=default_cfg)
        >>> result = ('cfg_lbl = %s' % (six.text_type(cfg_lbl),))
        >>> print(result)
        cfg_lbl = :var2=val2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfg = {'_cfgname': 'test:K=[1,2,3]', 'K': '1'}
        >>> name = None
        >>> nonlbl_keys = ['_cfgstr', '_cfgname', '_cfgtype', '_cfgindex']
        >>> cfg_lbl = get_cfg_lbl(cfg, name, nonlbl_keys)
        >>> result = ('cfg_lbl = %s' % (six.text_type(cfg_lbl),))
        >>> print(result)
        cfg_lbl = test:K=1
    """
    import utool as ut
    if name is None:
        name = cfg.get('_cfgname', '')

    if default_cfg is not None:
        # Remove defaulted labels
        cfg = ut.partition_varied_cfg_list([cfg], default_cfg)[1][0]

    # remove keys that should not belong to the label
    _clean_cfg = ut.delete_keys(cfg.copy(), nonlbl_keys)
    _lbl = ut.repr4(_clean_cfg, explicit=True, nl=False, strvals=True,
                       key_order=key_order, itemsep=sep)
    # _search = ['dict(', ')', ' ']
    _search = ['dict(', ')']
    _repl = [''] * len(_search)
    _lbl = ut.multi_replace(_lbl, _search, _repl).rstrip(',')
    if not with_name:
        return _lbl
    if NAMEVARSEP in name:
        # hack for when name contains a little bit of the _lbl
        # VERY HACKY TO PARSE OUT PARTS OF THE GIVEN NAME.
        hacked_name, _cfgstr, _ = parse_cfgstr_name_options(name)
        _cfgstr_options_list = re.split(
            r',\s*' + ut.negative_lookahead(r'[^\[\]]*\]'), _cfgstr)
        #cfgstr_options_list = cfgopt_strs.split(',')
        _cfg_options = ut.parse_cfgstr_list(
            _cfgstr_options_list, smartcast=False, oldmode=False)
        #
        ut.delete_keys(_cfg_options, cfg.keys())
        _preflbl = ut.repr4(_cfg_options, explicit=True, nl=False, strvals=True)
        _preflbl = ut.multi_replace(_preflbl, _search, _repl).rstrip(',')
        hacked_name += NAMEVARSEP + _preflbl
        ###
        cfg_lbl = hacked_name + _lbl
    else:
        cfg_lbl = name + NAMEVARSEP + _lbl
    return cfg_lbl


def recombine_nestings(parsed_blocks):
    import utool as ut  # NOQA
    if len(parsed_blocks) == 0:
        return ''
    values = ut.take_column(parsed_blocks, 1)
    literals = [recombine_nestings(v) if isinstance(v, list) else v for v in values]
    recombined = ''.join(literals)
    return recombined


class Nesting(object):
    r"""
    x = Nesting.from_nestings(nestings)
    list(x.itertype('nonNested'))
    """
    def __init__(self, type_, value):
        self.type_ = type_
        self.value = value

    def __repr__(self):
        return 'N' + repr(self.value)
        # if isinstance(self.value, list):
        #     # return str(self.type_) + ':' + repr(self.value)
        # else:
        #     return repr(self.value)

    def as_list(self):
        if self.type_ == 'root':
            if isinstance(self.value, list):
                return [x.as_list() for x in self.value]
        else:
            if isinstance(self.value, list):
                return (self.type_, [x.as_list() for x in self.value])
            elif self.value.__class__.__name__.endswith('Nesting'):
                return (self.type_, self.value.as_list())
            else:
                return (self.type_, self.value)

    def recombine(self):
        return recombine_nestings(self.as_list())

    @classmethod
    def from_text(cls, text):
        return cls.from_nestings(parse_nestings(text))

    @classmethod
    def from_nestings(cls, nestings, type_='root'):
        if isinstance(nestings, list):
            if type_ == 'root':
                return cls(type_, [cls.from_nestings(v) for v in nestings])
            else:
                return [cls.from_nestings(v) for v in nestings]
        elif isinstance(nestings, tuple):
            type_, v = nestings
            return cls(type_, cls.from_nestings(v, type_))
        else:
            return nestings

    def itertype(self, type_):
        if self.type_ == type_:
            yield self
        elif isinstance(self.value, list):
            for v in self.value:
                for x in v.itertype(type_):
                    yield x


def parse_nestings2(string, nesters=['()', '[]', '<>', "''", '""'], escape='\\'):
    r"""
    References:
        http://stackoverflow.com/questions/4801403/pyparsing-nested-mutiple-opener-clo

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> string = r'1, 2, 3'
        >>> parsed_blocks = parse_nestings2(string)
        >>> print('parsed_blocks = {!r}'.format(parsed_blocks))

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> string = r'lambda u: sign(u) * abs(u)**3.0 * greater(u, 0)'
        >>> parsed_blocks = parse_nestings2(string)
        >>> print('parsed_blocks = {!r}'.format(parsed_blocks))
        >>> string = r'lambda u: sign("\"u(\'fdfds\')") * abs(u)**3.0 * greater(u, 0)'
        >>> parsed_blocks = parse_nestings2(string)
        >>> print('parsed_blocks = {!r}'.format(parsed_blocks))
        >>> recombined = recombine_nestings(parsed_blocks)
        >>> print('PARSED_BLOCKS = ' + ut.repr3(parsed_blocks, nl=1))
        >>> print('recombined = %r' % (recombined,))
        >>> print('orig       = %r' % (string,))
    """
    import utool as ut  # NOQA
    import pyparsing as pp

    def as_tagged(parent, doctag=None):
        """Returns the parse results as XML. Tags are created for tokens and lists that have defined results names."""
        # print('parent.__dict__ = {!r}'.format(parent.__dict__))
        try:
            toklist = parent._toklist
            tokdict = parent._tokdict
            if tokdict == '' and toklist == '':
                raise Exception  # python 2 compat
        except Exception:
            tokdict = parent._ParseResults__tokdict
            toklist = parent._ParseResults__toklist
        namedItems = dict((v[1], k) for (k, vlist) in tokdict.items()
                          for v in vlist)
        # collapse out indents if formatting is not desired
        parentTag = None
        if doctag is not None:
            parentTag = doctag
        else:
            if parent._ParseResults__name:
                parentTag = parent._ParseResults__name
        if not parentTag:
            parentTag = "ITEM"
        out = []
        for i, res in enumerate(toklist):
            if isinstance(res, pp.ParseResults):
                if i in namedItems:
                    child = as_tagged(res, namedItems[i])
                else:
                    child = as_tagged(res, None)
                out.append(child)
            else:
                # individual token, see if there is a name for it
                resTag = None
                if i in namedItems:
                    resTag = namedItems[i]
                if not resTag:
                    resTag = "ITEM"
                try:
                    child = (resTag, pp._ustr(res))
                except AttributeError:
                    child = (resTag, str(res))
                out += [child]
        return (parentTag, out)

    def combine_nested(opener, closer, content, name=None):
        r"""
        opener, closer, content = '(', ')', nest_body
        """
        import utool as ut  # NOQA
        ret1 = pp.Forward()
        # if opener == closer:
        #     closer = pp.Regex('(?<!' + re.escape(closer) + ')')
        _NEST = ut.identity
        #_NEST = pp.Suppress
        opener_ = _NEST(opener)
        closer_ = _NEST(closer)

        group = pp.Group(opener_ + pp.ZeroOrMore(content) + closer_)
        ret2 = ret1 << group
        if ret2 is None:
            ret2 = ret1
        else:
            pass
            #raise AssertionError('Weird pyparsing behavior. Comment this line if encountered. pp.__version__ = %r' % (pp.__version__,))
        if name is None:
            ret3 = ret2
        else:
            ret3 = ret2.setResultsName(name)
        assert ret3 is not None, 'cannot have a None return'
        return ret3

    # Current Best Grammar
    nest_body = pp.Forward()
    nest_expr_list = []
    for left, right in nesters:
        if left == right:
            # Treat left==right nestings as quoted strings
            q = left
            quotedString = pp.Group(q + pp.Regex(r'(?:[^{q}\n\r\\]|(?:{q}{q})|(?:\\(?:[^x]|x[0-9a-fA-F]+)))*'.format(q=q)) + q)
            nest_expr = quotedString.setResultsName('nest' + left + right)
        else:
            nest_expr = combine_nested(left, right, content=nest_body, name='nest' + left + right)
        nest_expr_list.append(nest_expr)

    # quotedString = Combine(Regex(r'"(?:[^"\n\r\\]|(?:"")|(?:\\(?:[^x]|x[0-9a-fA-F]+)))*')+'"'|
    #                        Regex(r"'(?:[^'\n\r\\]|(?:'')|(?:\\(?:[^x]|x[0-9a-fA-F]+)))*")+"'").setName("quotedString using single or double quotes")

    nonBracePrintables = ''.join(c for c in pp.printables if c not in ''.join(nesters)) + ' '
    nonNested = pp.Word(nonBracePrintables).setResultsName('nonNested')
    # nonNested = (pp.Word(nonBracePrintables) | pp.quotedString).setResultsName('nonNested')
    nonNested = nonNested.leaveWhitespace()

    # if with_curl and not with_paren and not with_brak:
    nest_body_input = nonNested
    for nest_expr in nest_expr_list:
        nest_body_input = nest_body_input | nest_expr

    nest_body << nest_body_input

    nest_body = nest_body.leaveWhitespace()
    parser = pp.ZeroOrMore(nest_body)

    debug_ = ut.VERBOSE

    if len(string) > 0:
        tokens = parser.parseString(string)
        if debug_:
            print('string = %r' % (string,))
            print('tokens List: ' + ut.repr3(tokens.asList()))
            try:
                print('tokens XML: ' + tokens.asXML())
            except Exception:
                print('tokens Dict: ' + repr(tokens.as_dict()))
        parsed_blocks = as_tagged(tokens)[1]
        if debug_:
            print('PARSED_BLOCKS = ' + ut.repr3(parsed_blocks, nl=1))
    else:
        parsed_blocks = []
    return parsed_blocks


def parse_nestings(string, only_curl=False):
    r"""
    References:
        http://stackoverflow.com/questions/4801403/pyparsing-nested-mutiple-opener-clo

    CommandLine:
        python -m utool.util_gridsearch parse_nestings:1 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> string = r'lambda u: sign(u) * abs(u)**3.0 * greater(u, 0)'
        >>> parsed_blocks = parse_nestings(string)
        >>> recombined = recombine_nestings(parsed_blocks)
        >>> print('PARSED_BLOCKS = ' + ut.repr3(parsed_blocks, nl=1))
        >>> print('recombined = %r' % (recombined,))
        >>> print('orig       = %r' % (string,))
        PARSED_BLOCKS = [
            ('nonNested', 'lambda u: sign'),
            ('paren', [('ITEM', '('), ('nonNested', 'u'), ('ITEM', ')')]),
            ('nonNested', '* abs'),
            ('paren', [('ITEM', '('), ('nonNested', 'u'), ('ITEM', ')')]),
            ('nonNested', '**3.0 * greater'),
            ('paren', [('ITEM', '('), ('nonNested', 'u, 0'), ('ITEM', ')')]),
        ]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> string = r'\chapter{Identification \textbf{foobar} workflow}\label{chap:application}'
        >>> parsed_blocks = parse_nestings(string)
        >>> print('PARSED_BLOCKS = ' + ut.repr3(parsed_blocks, nl=1))
        PARSED_BLOCKS = [
            ('nonNested', '\\chapter'),
            ('curl', [('ITEM', '{'), ('nonNested', 'Identification \\textbf'), ('curl', [('ITEM', '{'), ('nonNested', 'foobar'), ('ITEM', '}')]), ('nonNested', 'workflow'), ('ITEM', '}')]),
            ('nonNested', '\\label'),
            ('curl', [('ITEM', '{'), ('nonNested', 'chap:application'), ('ITEM', '}')]),
        ]
    """
    import utool as ut  # NOQA
    import pyparsing as pp

    def as_tagged(parent, doctag=None):
        """Returns the parse results as XML. Tags are created for tokens and lists that have defined results names."""
        try:
            toklist = parent._toklist
            tokdict = parent._tokdict
            if tokdict == '' and toklist == '':
                raise Exception  # python 2 compat
        except Exception:
            tokdict = parent._ParseResults__tokdict
            toklist = parent._ParseResults__toklist
        namedItems = dict((v[1], k) for (k, vlist) in tokdict.items()
                          for v in vlist)
        # collapse out indents if formatting is not desired
        parentTag = None
        if doctag is not None:
            parentTag = doctag
        else:
            if parent._ParseResults__name:
                parentTag = parent._ParseResults__name
        if not parentTag:
            parentTag = "ITEM"
        out = []
        for i, res in enumerate(toklist):
            if isinstance(res, pp.ParseResults):
                if i in namedItems:
                    child = as_tagged(res, namedItems[i])
                else:
                    child = as_tagged(res, None)
                out.append(child)
            else:
                # individual token, see if there is a name for it
                resTag = None
                if i in namedItems:
                    resTag = namedItems[i]
                if not resTag:
                    resTag = "ITEM"
                try:
                    child = (resTag, pp._ustr(res))
                except AttributeError:
                    child = (resTag, str(res))
                out += [child]
        return (parentTag, out)

    def combine_nested(opener, closer, content, name=None):
        r"""
        opener, closer, content = '(', ')', nest_body
        """
        import utool as ut  # NOQA
        ret1 = pp.Forward()
        _NEST = ut.identity
        #_NEST = pp.Suppress
        opener_ = _NEST(opener)
        closer_ = _NEST(closer)
        group = pp.Group(opener_ + pp.ZeroOrMore(content) + closer_)
        ret2 = ret1 << group
        if ret2 is None:
            ret2 = ret1
        else:
            pass
            #raise AssertionError('Weird pyparsing behavior. Comment this line if encountered. pp.__version__ = %r' % (pp.__version__,))
        if name is None:
            ret3 = ret2
        else:
            ret3 = ret2.setResultsName(name)
        assert ret3 is not None, 'cannot have a None return'
        return ret3

    # Current Best Grammar
    nest_body = pp.Forward()
    nestedParens   = combine_nested('(', ')', content=nest_body, name='paren')
    nestedBrackets = combine_nested('[', ']', content=nest_body, name='brak')
    nestedCurlies  = combine_nested('{', '}', content=nest_body, name='curl')

    nonBracePrintables = ''.join(c for c in pp.printables if c not in '(){}[]') + ' '
    nonNested = pp.Word(nonBracePrintables).setResultsName('nonNested')
    nonNested = nonNested.leaveWhitespace()

    # if with_curl and not with_paren and not with_brak:
    if only_curl:
        # TODO figure out how to chain |
        nest_body << (nonNested | nestedCurlies)
    else:
        nest_body << (nonNested | nestedParens | nestedBrackets | nestedCurlies)

    nest_body = nest_body.leaveWhitespace()
    parser = pp.ZeroOrMore(nest_body)

    debug_ = ut.VERBOSE

    if len(string) > 0:
        tokens = parser.parseString(string)
        if debug_:
            print('string = %r' % (string,))
            print('tokens List: ' + ut.repr3(tokens.asList()))
            try:
                print('tokens XML: ' + tokens.asXML())
            except Exception:
                print('tokens Dict: ' + repr(tokens.as_dict()))
        parsed_blocks = as_tagged(tokens)[1]
        if debug_:
            print('PARSED_BLOCKS = ' + ut.repr3(parsed_blocks, nl=1))
    else:
        parsed_blocks = []
    return parsed_blocks


def parse_cfgstr3(string, debug=None):
    r"""
    http://stackoverflow.com/questions/4801403/how-can-i-use-pyparsing-to-parse-nested-expressions-that-have-mutiple-opener-clo

    Ignore:
        >>> from utool.util_gridsearch import *  # NOQA
        cfgopt_strs = 'f=2,c=[(1,2),(3,4)],d=1,j=[[1,2],[3,4]],foobar,x="[fdsfds",y="]fdsfds",e=[[1,2],[3,4]],[]'
        string  =  cfgopt_strs
        parse_cfgstr3(string)

    CommandLine:
        python -m utool.util_gridsearch parse_cfgstr3 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfgopt_strs = 'b=[1,2]'
        >>> cfgdict = parse_cfgstr3(cfgopt_strs, debug=0)
        >>> result = ('cfgdict = %s' % (ut.repr2(cfgdict),))
        >>> print(result)
        cfgdict = {'b': [1, 2]}

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfgopt_strs = 'myprefix=False,sentence_break=False'
        >>> cfgdict = parse_cfgstr3(cfgopt_strs, debug=0)
        >>> print('cfgopt_strs = %r' % (cfgopt_strs,))
        >>> result = ('cfgdict = %s' % (ut.repr2(cfgdict),))
        >>> print(result)
        cfgdict = {'myprefix': False, 'sentence_break': False}
    """
    import utool as ut  # NOQA
    import pyparsing as pp

    if debug is None:
        debug_ = ut.VERBOSE
    else:
        debug_ = debug

    def as_tagged(parent, doctag=None, namedItemsOnly=False):
        """Returns the parse results as XML. Tags are created for tokens and lists that have defined results names."""
        try:
            toklist = parent._toklist
            tokdict = parent._tokdict
            if tokdict == '' and toklist == '':
                raise Exception  # python 2 compat
        except Exception:
            tokdict = parent._ParseResults__tokdict
            toklist = parent._ParseResults__toklist
        namedItems = dict((v[1], k) for (k, vlist) in tokdict.items()
                          for v in vlist)
        # namedItems = dict((v[1], k) for (k, vlist) in parent._tokdict.items()
        #                   for v in vlist)
        # collapse out indents if formatting is not desire
        parentTag = None
        if doctag is not None:
            parentTag = doctag
        else:
            if parent._ParseResults__name:
                parentTag = parent._ParseResults__name
        if not parentTag:
            if namedItemsOnly:
                return ""
            else:
                parentTag = "ITEM"
        out = []
        for i, res in enumerate(toklist):
            if isinstance(res, pp.ParseResults):
                if i in namedItems:
                    child = as_tagged(
                        res, namedItems[i], namedItemsOnly and doctag is None)
                else:
                    child = as_tagged(
                        res, None, namedItemsOnly and doctag is None)
                out.append(child)
            else:
                # individual token, see if there is a name for it
                resTag = None
                if i in namedItems:
                    resTag = namedItems[i]
                if not resTag:
                    if namedItemsOnly:
                        continue
                    else:
                        resTag = "ITEM"
                try:
                    child = (resTag, pp._ustr(res))
                except AttributeError:
                    child = (resTag, str(res))
                out += [child]
        return (parentTag, out)

    def combine_nested(opener, closer, content, name=None):
        """
        opener, closer, content = '(', ')', nest_body
        """
        import utool as ut  # NOQA
        ret1 = pp.Forward()
        _NEST = ut.identity
        #_NEST = pp.Suppress
        opener_ = _NEST(opener)
        closer_ = _NEST(closer)
        # ret1 <<= pp.Group(opener_ + pp.ZeroOrMore(content) + closer_)
        ret2 = ret1 << pp.Group(opener_ + pp.ZeroOrMore(content) + closer_)
        if ret2 is None:
            ret2 = ret1
        else:
            pass
            #raise AssertionError('Weird pyparsing behavior. Comment this line if encountered. pp.__version__ = %r' % (pp.__version__,))
        if name is None:
            ret3 = ret2
        else:
            ret3 = ret2.setResultsName(name)
        assert ret3 is not None, 'cannot have a None return'
        return ret3

    # Current Best Grammar
    STRING = (pp.quotedString.copy()).setResultsName('quotedstring')
    NUM    = pp.Word(pp.nums).setResultsName('num')
    NAME   = pp.Regex('[a-zA-Z_][a-zA-Z_0-9]*')
    atom   = (NAME | NUM | STRING).setResultsName('atom')
    key    = pp.Word(pp.alphanums + '_').setResultsName('key')  # identifier

    nest_body = pp.Forward().setResultsName('nest_body')
    nestedParens   = combine_nested('(', ')', content=nest_body, name='paren')
    nestedBrackets = combine_nested('[', ']', content=nest_body, name='brak')
    nestedCurlies  = combine_nested('{', '}', content=nest_body, name='curl')

    nest_stmt = pp.Combine((nestedParens | nestedBrackets | nestedCurlies).setResultsName('sequence'))

    val = (atom | nest_stmt)

    # Nest body cannot have assignments in it
    #COMMA = pp.Suppress(',')
    COMMA = ','
    nest_body << val + pp.ZeroOrMore(COMMA + val)

    assign = pp.Group(key + pp.Suppress('=') + (val)).setResultsName('assign')
    #item = (assign | val).setResultsName('item')
    item = (assign | val)
    # OMG THIS LINE CAUSES NON-DETERMENISTIC RESULTS IN PYTHON3
    #item = (assign | val).setResultsName('item')

    # Assignments only allowed at outer level
    assign_body = item + pp.ZeroOrMore(pp.Suppress(',') + item)

    if len(string) > 0:
        tokens = assign_body.parseString(string)
        if debug_:
            print('string = %r' % (string,))
            print('tokens List: ' + ut.repr3(tokens.asList()))
            try:
                print('tokens XML: ' + tokens.asXML())
            except Exception:
                print('tokens Dict: ' + repr(tokens.as_dict()))
        parsed_blocks = as_tagged(tokens)[1]
        if debug_:
            print('PARSED_BLOCKS = ' + ut.repr3(parsed_blocks, nl=1))
    else:
        parsed_blocks = []

    from utool import util_type  # NOQA
    #from collections import OrderedDict
    #cfgdict = OrderedDict()
    cfgdict = {}
    for item in parsed_blocks:
        if item[0] == 'assign':
            keyval_pair = item[1]
            keytup, valtup = keyval_pair
            key = keytup[1]
            val = util_type.smart_cast2(valtup[1])
            #val = valtup[1]
        elif item[0] == 'atom':
            key = item[1]
            val = True
        else:
            key = None
            val = None
        if key is not None:
            cfgdict[key] = val
    if debug_:
        print('[TOKENS] CFGDICT ' + ut.repr3(cfgdict, nl=1))
        #x = list(map(type, cfgdict.keys()))
        #print(x)
        #print(list(cfgdict.keys()))
        #if len(x) > 0 and str(x[0]).find('Word') > -1:
        #    import utool
        #    utool.embed()
    return cfgdict


def noexpand_parse_cfgstrs(cfgopt_strs, alias_keys=None):
    r"""
    alias_keys = None
    cfgopt_strs = 'f=2,c=[(1,2),(3,4)],d=1'
    string  = cfgopt_strs

    cfgopt_strs = 'f=2,c=[1,2,3,4]'

    CommandLine:
        python -m utool.util_gridsearch noexpand_parse_cfgstrs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfgopt_strs = 'b=[1,2]'
        >>> alias_keys = None
        >>> cfg_combo = noexpand_parse_cfgstrs(cfgopt_strs, alias_keys)
        >>> result = ('cfg_combo = %s' % (ut.repr2(cfg_combo, nl=0),))
        >>> print(result)
        cfg_combo = {'b': [1, 2]}
    """
    import re
    import utool as ut
    # Parse dict out of a string
    #ANYTHING_NOT_BRACE = r'[^\[\]]*\]'
    #NOT_PAREN_OR_BRACE = r'[^()\[\]]*'
    if True:
        # Use a proper parser
        cfg_options = parse_cfgstr3(cfgopt_strs)
    else:
        not_paren_or_brace = '[^' + re.escape('()[]') + ']'
        end_paren_or_brace = '[' + re.escape(')]') + ']'
        nested_pat = not_paren_or_brace + '*' + end_paren_or_brace
        split_pat = r',\s*' + ut.negative_lookahead(nested_pat)
        cfgstr_options_list = re.split(split_pat, cfgopt_strs)

        cfg_options = ut.parse_cfgstr_list(
            cfgstr_list=cfgstr_options_list, smartcast=True, oldmode=False)
    # Remap keynames based on aliases
    if alias_keys is not None:
        # Use new standard keys and remove old aliased keys
        for key in set(alias_keys.keys()):
            if key in cfg_options:
                cfg_options[alias_keys[key]] = cfg_options[key]
                del cfg_options[key]
    return cfg_options


@util_decor.on_exception_report_input(keys=['cfgname', 'cfgopt_strs', 'base_cfg',
                                            'cfgtype', 'alias_keys', 'valid_keys'],
                                      force=True)
def customize_base_cfg(cfgname, cfgopt_strs, base_cfg, cfgtype,
                       alias_keys=None, valid_keys=None, offset=0,
                       strict=True):
    """
    Args:
        cfgname (str): config name
        cfgopt_strs (str): mini-language defining key variations
        base_cfg (dict): specifies the default cfg to customize
        cfgtype (?):
        alias_keys (None): (default = None)
        valid_keys (None): if base_cfg is not specied, this defines the valid
            keys (default = None)
        offset (int): (default = 0)
        strict (bool): (default = True)

    Returns:
        list: cfg_combo - list of config dicts defining customized configs
            based on cfgopt_strs. customized configs always are given an
            _cfgindex, _cfgstr, and _cfgname key.

    CommandLine:
        python -m utool.util_gridsearch --test-customize_base_cfg:0


    Ignore:
        >>> cfgname = 'default'
        >>> cfgopt_strs = 'dsize=1000,per_name=[1,2]'

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfgname = 'name'
        >>> cfgopt_strs = 'b=[1,2]'
        >>> base_cfg = {}
        >>> alias_keys = None
        >>> cfgtype = None
        >>> offset = 0
        >>> valid_keys = None
        >>> strict = False
        >>> cfg_combo = customize_base_cfg(cfgname, cfgopt_strs, base_cfg, cfgtype,
        >>>                                alias_keys, valid_keys, offset, strict)
        >>> result = ('cfg_combo = %s' % (ut.repr2(cfg_combo, nl=1),))
        >>> print(result)
        cfg_combo = [
            {'_cfgindex': 0, '_cfgname': 'name', '_cfgstr': 'name:b=[1,2]', '_cfgtype': None, 'b': 1},
            {'_cfgindex': 1, '_cfgname': 'name', '_cfgstr': 'name:b=[1,2]', '_cfgtype': None, 'b': 2},
        ]
    """
    import utool as ut
    cfg = base_cfg.copy()
    # Parse config options without expansion
    cfg_options = noexpand_parse_cfgstrs(cfgopt_strs, alias_keys)
    # Ensure that nothing bad is being updated
    if strict:
        parsed_keys = cfg_options.keys()
        if valid_keys is not None:
            ut.assert_all_in(parsed_keys, valid_keys,
                             'keys specified not in valid set')
        else:
            ut.assert_all_in(parsed_keys, cfg.keys(),
                             'keys specified not in default options')
    # Finalize configuration dict
    cfg.update(cfg_options)
    cfg['_cfgtype'] = cfgtype
    cfg['_cfgname'] = cfgname
    # Perform expansion
    cfg_combo = ut.all_dict_combinations(cfg)
    #if len(cfg_combo) > 1:
    for combox, cfg_ in enumerate(cfg_combo, start=offset):
        cfg_['_cfgindex'] = combox
    for cfg_ in cfg_combo:
        if len(cfgopt_strs) > 0:
            cfg_['_cfgstr'] = cfg_['_cfgname'] + NAMEVARSEP + cfgopt_strs
        else:
            cfg_['_cfgstr'] = cfg_['_cfgname']
    return cfg_combo


def parse_cfgstr_name_options(cfgstr):
    r"""
    Args:
        cfgstr (str):

    Returns:
        tuple: (cfgname, cfgopt_strs, subx)

    CommandLine:
        python -m utool.util_gridsearch --test-parse_cfgstr_name_options

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfgstr = 'default' + NAMEVARSEP + 'myvar1=myval1,myvar2=myval2'
        >>> (cfgname, cfgopt_strs, subx) = parse_cfgstr_name_options(cfgstr)
        >>> result = ('(cfgname, cfg_optstrs, subx) = %s' % (ut.repr2((cfgname, cfgopt_strs, subx)),))
        >>> print(result)
        (cfgname, cfg_optstrs, subx) = ('default', 'myvar1=myval1,myvar2=myval2', None)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfgstr = 'default[0:1]' + NAMEVARSEP + 'myvar1=myval1,myvar2=myval2'
        >>> (cfgname, cfgopt_strs, subx) = parse_cfgstr_name_options(cfgstr)
        >>> result = ('(cfgname, cfg_optstrs, subx) = %s' % (ut.repr2((cfgname, cfgopt_strs, subx)),))
        >>> print(result)
        (cfgname, cfg_optstrs, subx) = ('default', 'myvar1=myval1,myvar2=myval2', slice(0, 1, None))

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> cfgstr = 'default[0]' + NAMEVARSEP + 'myvar1=myval1,myvar2=myval2'
        >>> (cfgname, cfgopt_strs, subx) = parse_cfgstr_name_options(cfgstr)
        >>> result = ('(cfgname, cfg_optstrs, subx) = %s' % (ut.repr2((cfgname, cfgopt_strs, subx)),))
        >>> print(result)
        (cfgname, cfg_optstrs, subx) = ('default', 'myvar1=myval1,myvar2=myval2', [0])
    """
    import utool as ut
    cfgname_regex = ut.named_field('cfgname', r'[^\[:]*')  # name is optional
    subx_regex = r'\[' + ut.named_field('subx', r'[^\]]*') + r'\]'
    cfgopt_regex = re.escape(NAMEVARSEP) + ut.named_field('cfgopt', '.*')
    regex_str = cfgname_regex + ('(%s)?' % (subx_regex,)) + ('(%s)?' % (cfgopt_regex,))
    match = re.match(regex_str, cfgstr)
    assert match is not None, 'parsing of cfgstr failed'
    groupdict = match.groupdict()
    cfgname = groupdict['cfgname']
    if cfgname == '':
        cfgname = 'default'
    cfgopt_strs = groupdict.get('cfgopt', None)
    subx_str = groupdict.get('subx', None)
    if cfgopt_strs is None:
        cfgopt_strs = ''
    subx = ut.fuzzy_subset(subx_str)
    return cfgname, cfgopt_strs, subx


def lookup_base_cfg_list(cfgname, named_defaults_dict, metadata=None):
    import utool as ut
    if named_defaults_dict is None:
        base_cfg_list = [{}]
    else:
        try:
            base_cfg_list = named_defaults_dict[cfgname]
        except KeyError as ex:
            ut.printex(ex, 'Unknown configuration name', keys=['cfgname'])
            raise
    if ut.is_funclike(base_cfg_list):
        # make callable function
        base_cfg_list = base_cfg_list(metadata)
    if not isinstance(base_cfg_list, list):
        base_cfg_list = [base_cfg_list]
    return base_cfg_list


def parse_cfgstr_list2(cfgstr_list, named_defaults_dict=None, cfgtype=None,
                       alias_keys=None, valid_keys=None, expand_nested=True,
                       strict=True, special_join_dict=None, is_nestedcfgtype=False,
                       metadata=None):
    r"""
    Parses config strings. By looking up name in a dict of configs

    Args:
        cfgstr_list (list):
        named_defaults_dict (dict): (default = None)
        cfgtype (None): (default = None)
        alias_keys (None): (default = None)
        valid_keys (None): (default = None)
        expand_nested (bool): (default = True)
        strict (bool): (default = True)
        is_nestedcfgtype - used for annot configs so special joins arent geometrically combined

    Note:
        Normal Case:
            --flag name

        Custom Arugment Cases:
            --flag name:custom_key1=custom_val1,custom_key2=custom_val2

        Multiple Config Case:
            --flag name1:custom_args1 name2:custom_args2

        Multiple Config (special join) Case:
            (here name2 and name3 have some special interaction)
            --flag name1:custom_args1 name2:custom_args2::name3:custom_args3

        Varied Argument Case:
            --flag name:key1=[val1,val2]

    Returns:
        list: cfg_combos_list

    CommandLine:
        python -m utool.util_gridsearch --test-parse_cfgstr_list2
        python -m utool.util_gridsearch --test-parse_cfgstr_list2:0
        python -m utool.util_gridsearch --test-parse_cfgstr_list2:1
        python -m utool.util_gridsearch --test-parse_cfgstr_list2:2

    Setup:
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> named_defaults_dict = None
        >>> cfgtype, alias_keys, valid_keys, metadata = None, None, None, None
        >>> expand_nested, is_nestedcfgtypel, strict = True, False, False

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> named_defaults_dict = None
        >>> cfgtype, alias_keys, valid_keys, metadata = None, None, None, None
        >>> expand_nested, is_nestedcfgtypel, strict = True, False, False
        >>> cfgstr_list = ['name', 'name:f=1', 'name:b=[1,2]', 'name1:f=1::name2:f=1,b=2']
        >>> #cfgstr_list = ['name', 'name1:f=1::name2:f=1,b=2']
        >>> special_join_dict = {'joined': True}
        >>> cfg_combos_list = parse_cfgstr_list2(
        >>>     cfgstr_list, named_defaults_dict, cfgtype, alias_keys, valid_keys,
        >>>     expand_nested, strict, special_join_dict)
        >>> print('b' in cfg_combos_list[2][0])
        >>> print('cfg_combos_list = %s' % (ut.repr4(cfg_combos_list, nl=2),))
        >>> assert 'b' in cfg_combos_list[2][0], 'second cfg[2] should vary b'
        >>> assert 'b' in cfg_combos_list[2][1], 'second cfg[2] should vary b'
        >>> print(ut.depth_profile(cfg_combos_list))
        >>> cfg_list = ut.flatten(cfg_combos_list)
        >>> cfg_list = ut.flatten([cfg if isinstance(cfg, list) else [cfg] for cfg in cfg_list])
        >>> result = ut.repr2(ut.get_varied_cfg_lbls(cfg_list))
        >>> print(result)
        ['name:', 'name:f=1', 'name:b=1', 'name:b=2', 'name1:f=1,joined=True', 'name2:b=2,f=1,joined=True']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> # Allow for definition of a named default on the fly
        >>> cfgstr_list = ['base=:f=2,c=[1,2]', 'base:f=1', 'base:b=[1,2]']
        >>> special_join_dict = None
        >>> cfg_combos_list = parse_cfgstr_list2(
        >>>     cfgstr_list, named_defaults_dict, cfgtype, alias_keys, valid_keys,
        >>>     expand_nested, strict, special_join_dict)
        >>> print('cfg_combos_list = %s' % (ut.repr4(cfg_combos_list, nl=2),))
        >>> print(ut.depth_profile(cfg_combos_list))
        >>> cfg_list = ut.flatten(cfg_combos_list)
        >>> cfg_list = ut.flatten([cfg if isinstance(cfg, list) else [cfg] for cfg in cfg_list])
        >>> result = ut.repr2(ut.get_varied_cfg_lbls(cfg_list))
        >>> print(result)
        ['base:c=1,f=1', 'base:c=2,f=1', 'base:b=1,c=1,f=2', 'base:b=1,c=2,f=2', 'base:b=2,c=1,f=2', 'base:b=2,c=2,f=2']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> cfgstr_list = ['base:f=2,c=[(1,2),(3,4)]']
        >>> special_join_dict = None
        >>> cfg_combos_list = parse_cfgstr_list2(
        >>>     cfgstr_list, named_defaults_dict, cfgtype, alias_keys, valid_keys,
        >>>     expand_nested, strict, special_join_dict)
        >>> print('cfg_combos_list = %s' % (ut.repr4(cfg_combos_list, nl=2),))
        >>> print(ut.depth_profile(cfg_combos_list))
        >>> cfg_list = ut.flatten(cfg_combos_list)
        >>> cfg_list = ut.flatten([cfg if isinstance(cfg, list) else [cfg] for cfg in cfg_list])
        >>> result = ut.repr2(ut.get_varied_cfg_lbls(cfg_list))
        >>> print(result)

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> named_defaults_dict = None
        >>> cfgtype, alias_keys, valid_keys, metadata = None, None, None, None
        >>> expand_nested, is_nestedcfgtypel, strict = True, False, False
        >>> # test simplest case
        >>> cfgstr_list = ['name:b=[1,2]']
        >>> special_join_dict = {'joined': True}
        >>> cfg_combos_list = parse_cfgstr_list2(
        >>>     cfgstr_list, named_defaults_dict, cfgtype, alias_keys, valid_keys,
        >>>     expand_nested, strict, special_join_dict)
        >>> print('b' in cfg_combos_list[0][0])
        >>> print('cfg_combos_list = %s' % (ut.repr4(cfg_combos_list, nl=2),))
        >>> assert 'b' in cfg_combos_list[0][0], 'second cfg[2] should vary b'
        >>> assert 'b' in cfg_combos_list[0][1], 'second cfg[2] should vary b'
        >>> print(ut.depth_profile(cfg_combos_list))
        >>> cfg_list = ut.flatten(cfg_combos_list)
        >>> cfg_list = ut.flatten([cfg if isinstance(cfg, list) else [cfg] for cfg in cfg_list])
        >>> result = ut.repr2(ut.get_varied_cfg_lbls(cfg_list))
        >>> print(result)
    """
    import utool as ut
    #with ut.Indenter('    '):
    cfg_combos_list = []
    cfgstr_list_ = []

    # special named defaults assignment
    dyndef_named_defaults = {}
    for cfgstr in cfgstr_list:
        if cfgstr.find('=:') > -1:
            cfgname, cfgopt_strs, subx = parse_cfgstr_name_options(cfgstr)
            assert cfgname.endswith('=')
            cfgname = cfgname[:-1]
            base_cfg_list = lookup_base_cfg_list(cfgname,
                                                 named_defaults_dict,
                                                 metadata=metadata)
            cfg_options = noexpand_parse_cfgstrs(cfgopt_strs)
            dyndef_named_defaults[cfgname] = cfg_options
        else:
            cfgstr_list_.append(cfgstr)
    if len(dyndef_named_defaults) > 0 and named_defaults_dict is None:
        named_defaults_dict = dyndef_named_defaults

    for cfgstr in cfgstr_list_:
        cfg_combos = []
        # Parse special joined cfg case
        if cfgstr.find('::') > -1:
            special_cfgstr_list = cfgstr.split('::')
            # Recursive call
            special_combo_list = parse_cfgstr_list2(
                special_cfgstr_list,
                named_defaults_dict=named_defaults_dict, cfgtype=cfgtype,
                alias_keys=alias_keys, valid_keys=valid_keys,
                strict=strict, expand_nested=expand_nested,
                is_nestedcfgtype=False, metadata=metadata)
            if special_join_dict is not None:
                for special_combo in special_combo_list:
                    for cfg in special_combo:
                        cfg.update(special_join_dict)
            if is_nestedcfgtype:
                cfg_combo = tuple([combo for combo in special_combo_list])
            else:
                # not sure if this is right
                cfg_combo = special_combo_list
            # FIXME DUPLICATE CODE
            if expand_nested:
                cfg_combos.extend(cfg_combo)
            else:
                #print('Appending: ' + str(ut.depth_profile(cfg_combo)))
                #if ut.depth_profile(cfg_combo) == [1, 9]:
                #    ut.embed()
                cfg_combos_list.append(cfg_combo)
        else:
            # Normal Case
            cfgname, cfgopt_strs, subx = parse_cfgstr_name_options(cfgstr)
            # --
            # Lookup named default settings
            try:
                base_cfg_list = lookup_base_cfg_list(cfgname,
                                                     named_defaults_dict,
                                                     metadata=metadata)
            except Exception as ex:
                ut.printex(ex, keys=['cfgstr_list', 'cfgstr_list_'])
                raise
            # --
            for base_cfg in base_cfg_list:
                print('cfgname = %r' % (cfgname,))
                print('cfgopt_strs = %r' % (cfgopt_strs,))
                print('base_cfg = %r' % (base_cfg,))
                print('alias_keys = %r' % (alias_keys,))
                print('cfgtype = %r' % (cfgtype,))
                print('offset = %r' % (len(cfg_combos),))
                print('valid_keys = %r' % (valid_keys,))
                print('strict = %r' % (strict,))
                cfg_combo = customize_base_cfg(
                    cfgname, cfgopt_strs, base_cfg, cfgtype, alias_keys,
                    valid_keys, strict=strict, offset=len(cfg_combos))
                if is_nestedcfgtype:
                    cfg_combo = [cfg_combo]
                if expand_nested:
                    cfg_combos.extend(cfg_combo)
                else:
                    cfg_combos_list.append(cfg_combo)
        # SUBX Cannot work here because of acfg hackiness
        #if subx is not None:
        #    cfg_combo = ut.take(cfg_combo, subx)
        if expand_nested:
            cfg_combos_list.append(cfg_combos)
    #    print('Updated to: ' + str(ut.depth_profile(cfg_combos_list)))
    #print('Returning len(cfg_combos_list) = %r' % (len(cfg_combos_list),))
    return cfg_combos_list


@util_class.reloadable_class
class ParamInfo(util_dev.NiceRepr):
    """
    small class for individual paramater information

    Configuration objects should use these for default / printing / type
    information.
    NOTE: the actual value of the parameter for any specific configuration is
    not stored here.

    CommandLine:
        python -m utool.util_gridsearch --test-ParamInfo

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> pi = ParamInfo(varname='foo', default='bar')
        >>> cfg = ut.DynStruct()
        >>> cfg.foo = 5
        >>> result = pi.get_itemstr(cfg)
        >>> print(result)
        foo=5
    """
    def __init__(pi, varname=None, default=None, shortprefix=None,
                 type_=util_const.NoParam, varyvals=[], varyslice=None,
                 hideif=util_const.NoParam, help_=None, valid_values=None,
                 max_=None, min_=None, step_=None, none_ok=True):
        r"""
        Args:
            varname (str): name of the variable
            default (str): default value of the variable
            shortprefix (str): shorter version of the varname for the cfgstr
            type_ (type): python type of the variable (inferable)
            varyvals (list): set of values usable by gridsearch
            valid_values (list): set of allowed values
            hideif (func or value): if the variable value of config is this the
                itemstr is empty
        """
        pi.varname = varname
        pi.default = default
        pi.shortprefix = shortprefix
        if type_ is util_const.NoParam:
            if default is not None:
                pi.type_ = type(default)
            else:
                pi.type_ = None
        else:
            pi.type_ = type_
        # pi.type_ = type(default) if type_ is util_const.NoParam else type_
        # for gridsearch
        pi.varyvals = varyvals
        pi.varyslice = varyslice
        pi.none_ok = none_ok
        if valid_values is not None:
            if default not in valid_values:
                # hack
                valid_values.append(default)
            assert default in valid_values
        pi.valid_values = valid_values
        pi.hideif_list = []
        if hideif is not util_const.NoParam:
            pi.append_hideif(hideif)
        pi.max_ = max_
        pi.min_ = min_
        pi.step_ = step_

    def __nice__(pi):
        return pi._make_varstr(pi.default)

    def is_type_enforced(pi):
        enforce_type = (pi.type_ is not None and
                        not isinstance(pi.type_, six.string_types) and
                        isinstance(pi.type_, type))
        return enforce_type

    def error_if_invalid_value(pi, value):
        """
        Checks if a `value` for this param is valid
        """
        if pi.none_ok and value is None:
            return True
        if pi.valid_values is not None:
            if value not in pi.valid_values:
                raise ValueError('pi=%r, value=%r not in valid_values=%r' %
                                 (pi, value, pi.valid_values,))
        if pi.is_type_enforced():
            if not util_type.is_comparable_type(value, pi.type_):
                raise TypeError(
                    'pi=%r, value=%r with type_=%r is not the expected '
                    'type_=%r' % (pi, value, type(value), pi.type_))

    def cast_to_valid_type(pi, value):
        """
        Checks if a `value` for this param is valid and attempts
        to cast it if it is close
        """
        if pi.none_ok and value is None:
            return True
        if pi.valid_values is not None:
            if value not in pi.valid_values:
                raise ValueError('pi=%r, value=%r not in valid_values=%r' %
                                 (pi, value, pi.valid_values,))
        if not util_type.is_comparable_type(value, pi.type_):
            raise TypeError(
                'pi=%r, value=%r with type_=%r is not the expected type_=%r' %
                (pi, value, type(value), pi.type_))

    def append_hideif(pi, hideif):
        pi.hideif_list.append(hideif)

    def is_hidden(pi, cfg):
        """
        Checks if this param is hidden by the parent context of `cfg`
        """
        for hideif in pi.hideif_list:
            if callable(hideif):
                hide = hideif(cfg)
            elif (isinstance(hideif, six.string_types) and
                  hideif.startswith(':')):
                # advanced hideif parsing
                print('Checking hide for {}'.format(pi.varname))
                code = hideif[1:]
                filename = '<{}>.__hideif__'.format(pi.varname)
                evalable = compile(code, filename, 'eval')
                hide = eval(evalable, cfg)
                print('hide = %r' % (hide,))
            else:
                # just a value wrt to this param
                hide = getattr(cfg,  pi.varname) == hideif
            if hide:
                return True

    def is_enabled(pi, cfg):
        """
        alternative to is_hidden. a bit hacky. only returns false if
        another config hides you. self hiding is ok
        """
        enabled = True
        for hideif in pi.hideif_list:
            if callable(hideif):
                # FIXME: just because it is a function doesnt mean it isn't
                # self-hidden
                enabled = not hideif(cfg)
            if not enabled:
                break
        return enabled

    def _make_varstr(pi, varval):
        # Create string representation of a `varval`
        if isinstance(varval, set):
            import utool as ut
            varstr = 'set([' + ut.repr4(sorted(varval), nl=0, itemsep='', nobr=True) + '])'
        elif isinstance(varval, dict):
            from utool import util_str
            # The ut.repr4 representation will consistently stringify dicts
            varstr = util_str.repr4(varval, explicit=True, strvals=True,
                                       itemsep='', nl=0)
        else:
            varstr = six.text_type(varval)
            if isinstance(varval, slice):
                varstr = varstr.replace(' ', '')
        # Place varstr in the context of the param name
        if pi.shortprefix is not None:
            itemstr = '%s%s' % (pi.shortprefix, varstr)
        else:
            itemstr =  '%s=%s' % (pi.varname, varstr)
        return itemstr

    def _make_itemstr(pi, cfg):
        varval = getattr(cfg,  pi.varname)
        return pi._make_varstr(varval)

    def get_itemstr(pi, cfg):
        """
        Args:
            cfg (dict-like): Parent object (like a dtool.Config)
        """
        if pi.is_hidden(cfg):
            return ''
        else:
            return pi._make_itemstr(cfg)


@six.add_metaclass(util_class.ReloadingMetaclass)
class ParamInfoBool(ParamInfo):
    r"""
    param info for booleans

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> pi = ParamInfoBool('cheese_on', hideif=util_const.NoParam)
        >>> cfg = ut.DynStruct()
        >>> cfg.cheese_on = False
        >>> result = pi.get_itemstr(cfg)
        >>> print(result)
        nocheese
    """
    def __init__(pi, varname, default=False, shortprefix=None,
                 type_=bool, varyvals=[], varyslice=None, hideif=False,
                 help_=None):
        if not varname.endswith('_on'):
            # TODO: use this convention or come up with a better one
            #print('WARNING: varname=%r should end with _on' % (varname,))
            pass
        import utool as ut
        _ParamInfoBool = ut.fix_super_reload(ParamInfoBool, pi)
        super(_ParamInfoBool, pi).__init__(
            varname, default=default, shortprefix=shortprefix, type_=bool,
            varyvals=varyvals, varyslice=varyslice, hideif=hideif)

    def _make_itemstr(pi, cfg):
        # TODO: redo this as _make_varstr and remove all instances of
        # _make_itemstr
        varval = getattr(cfg,  pi.varname)
        if pi.shortprefix is not None:
            itemstr = pi.shortprefix
        else:
            itemstr =  pi.varname.replace('_on', '')
        if varval is False:
            itemstr = 'no' + itemstr
        elif varval is not True:
            raise AssertionError('Not a boolean pi.varname=%r, varval=%r' % (
                pi.varname, varval,))
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
        slice_dict = {pi.varname: pi.varyslice for pi in self.param_info_list
                      if pi.varyslice is not None}
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
        ...     tp_score = cfgdict['p'] + (cfgdict['K'] ** .5)
        ...     tn_score = (cfgdict['p'] * (cfgdict['K'])) / cfgdict['dcvs_clip_max']
        ...     gridsearch.append_result(tp_score, tn_score)
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
            param_label_list = ut.compress(param_label_list_, is_varied)
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
            >>> import plottool_ibeis as pt
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
            grid_basis_str=ut.repr4(gridsearch.grid_basis),
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
        #print(ut.repr4(param2_score_stats))
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
        param_stats_str = 'stats(' + param_lbl + ') = ' + ut.repr4(param2_score_stats_str)
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
            >>> import plottool_ibeis as pt
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
        import plottool_ibeis as pt
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
        ...     DimensionBasis('dim1', [.1, .2, .3]),
        ...     DimensionBasis('dim2', [.1, .4, .5]),
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
        >>> result = six.text_type((cfgdict_list, cfglbl_list))
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
        ...     {'K': 3, 'dcvs_clip_max': 0.1, 'p': 0.1},
        ...     {'K': 5, 'dcvs_clip_max': 0.1, 'p': 0.1},
        ...     {'K': 5, 'dcvs_clip_max': 0.1, 'p': 0.2},
        ...     {'K': 3, 'dcvs_clip_max': 0.2, 'p': 0.1},
        ...     {'K': 5, 'dcvs_clip_max': 0.2, 'p': 0.1},
        ...     {'K': 3, 'dcvs_clip_max': 0.2, 'p': 0.1}]
        >>> keys = ['K', 'dcvs_clip_max']
        >>> # execute function
        >>> cfgdict_sublist = get_cfgdict_list_subset(cfgdict_list, keys)
        >>> # verify results
        >>> result = ut.repr4(cfgdict_sublist)
        >>> print(result)
        [
            {'K': 3, 'dcvs_clip_max': 0.1},
            {'K': 5, 'dcvs_clip_max': 0.1},
            {'K': 3, 'dcvs_clip_max': 0.2},
            {'K': 5, 'dcvs_clip_max': 0.2},
        ]
    """
    import utool as ut
    cfgdict_sublist_ = [ut.dict_subset(cfgdict, keys) for cfgdict in cfgdict_list]
    cfgtups_sublist_ = [tuple(ut.dict_to_keyvals(cfgdict)) for cfgdict in cfgdict_sublist_]
    cfgtups_sublist = ut.unique_ordered(cfgtups_sublist_)
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
        cfglbl = six.text_type(cfgdict)
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
    assert callable(show_result_func), 'NEED FUNCTION GOT: %r' % (show_result_func,)

    import utool as ut
    import plottool_ibeis as pt
    from plottool_ibeis import plot_helpers as ph
    from plottool_ibeis import interact_helpers as ih
    if verbose:
        print('Plotting gridsearch results figtitle=%r' % (figtitle,))
    if score_list is None:
        score_list = [None] * len(cfgdict_list)
    else:
        # sort by score if available
        sortx_list = ut.list_argsort(score_list, reverse=True)
        score_list = ut.take(score_list, sortx_list)
        cfgdict_list = ut.take(cfgdict_list, sortx_list)
        cfglbl_list = ut.take(cfglbl_list, sortx_list)
        cfgresult_list = ut.take(cfgresult_list, sortx_list)
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
            cfglbl += '\n' + scorelbl + '=' + ut.repr2(score, precision=precision)
        pnum = next_pnum()
        try:
            if unpack:
                show_result_func(*cfgresult, fnum=fnum, pnum=pnum)
            else:
                show_result_func(cfgresult, fnum=fnum, pnum=pnum)
        except Exception as ex:
            if isinstance(cfgresult, tuple):
                #print(ut.repr4(cfgresult))
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
            print(ut.repr4(plotdat_dict))
            cfglbl = ph.get_plotdat(ax, 'cfglbl', None)
            cfgdict = ph.get_plotdat(ax, 'cfgdict', {})
            cfgresult = ph.get_plotdat(ax, 'cfgresult', {})
            infostr_list = [
                ('cfglbl = %s' % (cfglbl,)),
                '',
                ('cfgdict = ' + ut.repr4(cfgdict, sorted_=True)),
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


def gridsearch_timer(func_list, args_list, niters=None, **searchkw):
    """
    Times a series of functions on a series of inputs

    args_list is a list should vary the input sizes
    can also be a func that take a count param

    items in args_list list or returned by the func should be a tuple so it can be
    unpacked

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_rowids_from_aid2 --show
        python -m ibeis.annotmatch_funcs --exec-get_annotmatch_rowids_from_aid:1 --show

    Args:
        func_list (list):
        args_list (list):
        niters (None): (default = None)

    Returns:
        dict: time_result

    CommandLine:
        python -m utool.util_gridsearch --exec-gridsearch_timer --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_gridsearch import *  # NOQA
        >>> import utool as ut
        >>> func_list = [ut.fibonacci_recursive, ut.fibonacci_iterative]
        >>> args_list = list(range(1, 35))
        >>> niters = None
        >>> searchkw = {}
        >>> time_result = gridsearch_timer(func_list, args_list, niters, **searchkw)
        >>> result = ('time_result = %s' % (six.text_type(time_result),))
        >>> print(result)
        >>> time_result['plot_timings']()
        >>> ut.show_if_requested()
    """
    import utool as ut
    timings = ut.ddict(list)
    if niters is None:
        niters = len(args_list)

    if ut.is_funclike(args_list):
        get_args = args_list
    else:
        get_args = args_list.__getitem__

    #func_labels = searchkw.get('func_labels', list(range(len(func_list))))
    func_labels = searchkw.get('func_labels', [ut.get_funcname(func) for func in func_list])
    use_cache = searchkw.get('use_cache', not ut.get_argflag(('--nocache', '--nocache-time')))
    assert_eq = searchkw.get('assert_eq', True)

    count_list = list(range(niters))
    xlabel_list = []

    cache = ut.ShelfCacher('timeings.shelf', enabled=use_cache)

    for count in ut.ProgressIter(count_list, lbl='Testing Timeings'):
        args_ = get_args(count)
        xlabel_list.append(args_)
        if True:
            # HACK
            # There is an unhandled corner case that will fail if the function expects a tuple.
            if not isinstance(args_, tuple):
                args_ = (args_,)
        assert isinstance(args_, tuple), 'args_ should be a tuple so it can be unpacked'
        ret_list = []
        for func_ in func_list:
            try:
                kwargs_ = {}
                func_cachekey = ut.get_func_result_cachekey(func_, args_, kwargs_)
                ellapsed = cache.load(func_cachekey)
            except ut.CacheMissException:
                with ut.Timer(verbose=False) as t:
                    ret = func_(*args_)
                    ret_list.append(ret)
                ellapsed = t.ellapsed
                cache.save(func_cachekey, ellapsed)
            timings[func_].append(ellapsed)
        if assert_eq:
            # Hacky, not guarenteed to work if cache is one
            ut.assert_all_eq(list(map(ut.cachestr_repr, ret_list)))

    cache.close()

    count_to_xtick = searchkw.get('count_to_xtick', lambda x, y: x)
    xtick_list = [count_to_xtick(count, get_args(count)) for count in count_list]

    def plot_timings():
        import plottool_ibeis as pt
        ydata_list = ut.dict_take(timings, func_list)
        xdata = xtick_list

        ylabel = 'seconds'
        xlabel = 'input size'

        pt.multi_plot(
            xdata, ydata_list, label_list=func_labels,
            ylabel=ylabel, xlabel=xlabel,
            **searchkw
        )
    time_result = {
        'plot_timings': plot_timings,
        'timings': timings,
    }
    return time_result


if __name__ == '__main__':
    """
    pycallgraph graphviz -- ~/code/utool/utool/util_gridsearch.py

    CommandLine:
        python -m utool.util_gridsearch
        python -m utool.util_gridsearch --allexamples
        python -m utool.util_gridsearch --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
