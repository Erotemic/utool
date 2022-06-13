# -*- coding: utf-8 -*-
# TODO Licence:
#
# TODO:  move library intensive functions to vtool
from __future__ import absolute_import, division, print_function, unicode_literals
# import operator
import six
# from six.moves import zip, range, reduce
# from utool import util_type
from utool import util_inject
# from utool import util_decor
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    # TODO remove numpy
    pass
try:
    import scipy.spatial.distance as spdist  # NOQA
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
print, rrr, profile = util_inject.inject2(__name__)


unicode = six.text_type

PHI = 1.61803398875
PHI_A = (1 / PHI)
PHI_B = 1 - PHI_A


def bayesnet():
    """
    References:
        https://class.coursera.org/pgm-003/lecture/17
        http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
        http://www3.cs.stonybrook.edu/~sael/teaching/cse537/Slides/chapter14d_BP.pdf
        http://www.cse.unsw.edu.au/~cs9417ml/Bayes/Pages/PearlPropagation.html
        https://github.com/pgmpy/pgmpy.git
        http://pgmpy.readthedocs.org/en/latest/
        http://nipy.bic.berkeley.edu:5000/download/11
    """
    import utool as ut
    # import operator as op
    # # Enumerate all possible events
    # varcard_list = list(map(op.attrgetter('variable_card'), cpd_list))
    # _esdat = list(ut.iprod(*map(range, varcard_list)))
    # _escol = list(map(op.attrgetter('variable'), cpd_list))
    # event_space = pd.DataFrame(_esdat, columns=_escol)

    # # Custom compression of event space to inspect a specific graph
    # def compress_space_flags(event_space, var1, var2, var3, cmp12_):
    #     """
    #     var1, var2, cmp_ = 'Lj', 'Lk', op.eq
    #     """
    #     import vtool as vt
    #     data = event_space
    #     other_cols = ut.setdiff_ordered(data.columns.tolist(), [var1, var2, var3])
    #     case_flags12 = cmp12_(data[var1], data[var2]).values
    #     # case_flags23 = cmp23_(data[var2], data[var3]).values
    #     # case_flags = np.logical_and(case_flags12, case_flags23)
    #     case_flags = case_flags12
    #     case_flags = case_flags.astype(np.int64)
    #     subspace = np.hstack((case_flags[:, None], data[other_cols].values))
    #     sel_ = vt.unique_row_indexes(subspace)
    #     flags = np.logical_and(mask, case_flags)
    #     return flags

    # # Build special cases
    # case_same   = event_space.loc[compress_space_flags(event_space, 'Li', 'Lj', 'Lk', op.eq)]
    # case_diff = event_space.loc[compress_space_flags(event_space, 'Li', 'Lj', 'Lk', op.ne)]
    # special_cases = [
    #     case_same,
    #     case_diff,
    # ]

    from pgmpy.factors import TabularCPD
    from pgmpy.models import BayesianModel
    import pandas as pd
    from pgmpy.inference import BeliefPropagation  # NOQA
    from pgmpy.inference import VariableElimination  # NOQA

    name_nice = ['n1', 'n2', 'n3']
    score_nice = ['low', 'high']
    match_nice = ['diff', 'same']
    num_names = len(name_nice)
    num_scores = len(score_nice)
    nid_basis = list(range(num_names))
    score_basis = list(range(num_scores))

    semtype2_nice = {
        'score': score_nice,
        'name': name_nice,
        'match': match_nice,
    }
    var2_cpd = {
    }
    globals()['semtype2_nice'] = semtype2_nice
    globals()['var2_cpd'] = var2_cpd

    name_combo = np.array(list(ut.iprod(nid_basis, nid_basis)))
    combo_is_same = name_combo.T[0] == name_combo.T[1]
    def get_expected_scores_prob(level1, level2):
        part1 = combo_is_same * level1
        part2 = (1 - combo_is_same) * (1 - (level2))
        expected_scores_level = part1 + part2
        return expected_scores_level

    # def make_cpd():

    def name_cpd(aid):
        from pgmpy.factors import TabularCPD
        cpd = TabularCPD(
            variable='N' + aid,
            variable_card=num_names,
            values=[[1.0 / num_names] * num_names])
        cpd.semtype = 'name'
        return cpd

    name_cpds = [name_cpd('i'), name_cpd('j'), name_cpd('k')]
    var2_cpd.update(dict(zip([cpd.variable for cpd in name_cpds], name_cpds)))
    if True:
        num_same_diff = 2
        samediff_measure = np.array([
            # get_expected_scores_prob(.12, .2),
            # get_expected_scores_prob(.88, .8),
            get_expected_scores_prob(0, 0),
            get_expected_scores_prob(1, 1),
        ])
        samediff_vals = (samediff_measure / samediff_measure.sum(axis=0)).tolist()
        def samediff_cpd(aid1, aid2):
            cpd = TabularCPD(
                variable='A' + aid1 + aid2,
                variable_card=num_same_diff,
                values=samediff_vals,
                evidence=['N' + aid1, 'N' + aid2],  # [::-1],
                evidence_card=[num_names, num_names])  # [::-1])
            cpd.semtype = 'match'
            return cpd
        samediff_cpds = [samediff_cpd('i', 'j'), samediff_cpd('j', 'k'), samediff_cpd('k', 'i')]
        var2_cpd.update(dict(zip([cpd.variable for cpd in samediff_cpds], samediff_cpds)))

        if True:
            def score_cpd(aid1, aid2):
                semtype = 'score'
                evidence = ['A' + aid1 + aid2, 'N' + aid1, 'N' + aid2]
                evidence_cpds = [var2_cpd[key] for key in evidence]
                evidence_nice = [semtype2_nice[cpd.semtype] for cpd in evidence_cpds]
                evidence_card = list(map(len, evidence_nice))
                evidence_states = list(ut.iprod(*evidence_nice))
                variable_basis = semtype2_nice[semtype]

                variable_values = []
                for mystate in variable_basis:
                    row = []
                    for state in evidence_states:
                        if state[0] == state[1]:
                            if state[2] == 'same':
                                val = .2 if mystate == 'low' else .8
                            else:
                                val = 1
                                # val = .5 if mystate == 'low' else .5
                        elif state[0] != state[1]:
                            if state[2] == 'same':
                                val = .5 if mystate == 'low' else .5
                            else:
                                val = 1
                                # val = .9 if mystate == 'low' else .1
                        row.append(val)
                    variable_values.append(row)

                cpd = TabularCPD(
                    variable='S' + aid1 + aid2,
                    variable_card=len(variable_basis),
                    values=variable_values,
                    evidence=evidence,  # [::-1],
                    evidence_card=evidence_card)  # [::-1])
                cpd.semtype = semtype
                return cpd
        else:
            score_values = [
                [.8, .1],
                [.2, .9],
            ]
            def score_cpd(aid1, aid2):
                cpd = TabularCPD(
                    variable='S' + aid1 + aid2,
                    variable_card=num_scores,
                    values=score_values,
                    evidence=['A' + aid1 + aid2],  # [::-1],
                    evidence_card=[num_same_diff])  # [::-1])
                cpd.semtype = 'score'
                return cpd

        score_cpds = [score_cpd('i', 'j'), score_cpd('j', 'k')]
        cpd_list = name_cpds + score_cpds + samediff_cpds
    else:
        score_measure = np.array([get_expected_scores_prob(level1, level2)
                                  for level1, level2 in
                                  zip(np.linspace(.1, .9, num_scores),
                                      np.linspace(.2, .8, num_scores))])

        score_values = (score_measure / score_measure.sum(axis=0)).tolist()

        def score_cpd(aid1, aid2):
            cpd = TabularCPD(
                variable='S' + aid1 + aid2,
                variable_card=num_scores,
                values=score_values,
                evidence=['N' + aid1, 'N' + aid2],
                evidence_card=[num_names, num_names])
            cpd.semtype = 'score'
            return cpd
        score_cpds = [score_cpd('i', 'j'), score_cpd('j', 'k')]
        cpd_list = name_cpds + score_cpds
        pass

    input_graph = []
    for cpd in cpd_list:
        if cpd.evidence is not None:
            for evar in cpd.evidence:
                input_graph.append((evar, cpd.variable))
    name_model = BayesianModel(input_graph)
    name_model.add_cpds(*cpd_list)

    var2_cpd.update(dict(zip([cpd.variable for cpd in cpd_list], cpd_list)))
    globals()['var2_cpd'] = var2_cpd

    varnames = [cpd.variable for cpd in cpd_list]

    # --- PRINT CPDS ---

    cpd = score_cpds[0]
    def print_cpd(cpd):
        print('CPT: %r' % (cpd,))
        index = semtype2_nice[cpd.semtype]
        if cpd.evidence is None:
            columns = ['None']
        else:
            basis_lists = [semtype2_nice[var2_cpd[ename].semtype] for ename in cpd.evidence]
            columns = [','.join(x) for x in ut.iprod(*basis_lists)]
        data = cpd.get_cpd()
        print(pd.DataFrame(data, index=index, columns=columns))

    for cpd in name_model.get_cpds():
        print('----')
        print(cpd._str('phi'))
        print_cpd(cpd)

    # --- INFERENCE ---

    Ni = name_cpds[0]

    event_space_combos = {}
    event_space_combos[Ni.variable] = 0  # Set ni to always be Fred
    for cpd in cpd_list:
        if cpd.semtype == 'score':
            event_space_combos[cpd.variable] = list(range(cpd.variable_card))
    evidence_dict = ut.all_dict_combinations(event_space_combos)

    # Query about name of annotation k given different event space params

    def pretty_evidence(evidence):
        return [key + '=' + str(semtype2_nice[var2_cpd[key].semtype][val])
                for key, val in evidence.items()]

    def print_factor(factor):
        row_cards = factor.cardinality
        row_vars = factor.variables
        values = factor.values.reshape(np.prod(row_cards), 1).flatten()
        # col_cards = 1
        # col_vars = ['']
        basis_lists = list(zip(*list(ut.iprod(*[range(c) for c in row_cards]))))
        nice_basis_lists = []
        for varname, basis in zip(row_vars, basis_lists):
            cpd = var2_cpd[varname]
            _nice_basis = ut.take(semtype2_nice[cpd.semtype], basis)
            nice_basis = ['%s=%s' % (varname, val) for val in _nice_basis]
            nice_basis_lists.append(nice_basis)
        row_lbls = [', '.join(sorted(x)) for x in zip(*nice_basis_lists)]
        print(ut.repr3(dict(zip(row_lbls, values)), precision=3, align=True, key_order_metric='-val'))

    # name_belief = BeliefPropagation(name_model)
    name_belief = VariableElimination(name_model)
    import pgmpy
    import six  # NOQA

    def try_query(evidence):
        print('--------')
        query_vars = ut.setdiff_ordered(varnames, list(evidence.keys()))
        evidence_str = ', '.join(pretty_evidence(evidence))
        probs = name_belief.query(query_vars, evidence)
        factor_list = probs.values()
        joint_factor = pgmpy.factors.factor_product(*factor_list)
        print('P(' + ', '.join(query_vars) + ' | ' + evidence_str + ')')
        # print(six.text_type(joint_factor))
        factor = joint_factor  # NOQA
        # print_factor(factor)
        # import utool as ut
        print(ut.hz_str([(f._str(phi_or_p='phi')) for f in factor_list]))

    for evidence in evidence_dict:
        try_query(evidence)

    evidence = {'Aij': 1, 'Ajk': 1, 'Aki': 1, 'Ni': 0}
    try_query(evidence)

    evidence = {'Aij': 0, 'Ajk': 0, 'Aki': 0, 'Ni': 0}
    try_query(evidence)

    globals()['score_nice'] = score_nice
    globals()['name_nice'] = name_nice
    globals()['score_basis'] = score_basis
    globals()['nid_basis'] = nid_basis

    print('Independencies')
    print(name_model.get_independencies())
    print(name_model.local_independencies([Ni.variable]))

    # name_belief = BeliefPropagation(name_model)
    # # name_belief = VariableElimination(name_model)
    # for case in special_cases:
    #     test_data = case.drop('Lk', axis=1)
    #     test_data = test_data.reset_index(drop=True)
    #     print('----')
    #     for i in range(test_data.shape[0]):
    #         evidence = test_data.loc[i].to_dict()
    #         probs = name_belief.query(['Lk'], evidence)
    #         factor = probs['Lk']
    #         probs = factor.values
    #         evidence_ = evidence.copy()
    #         evidence_['Li'] = name_nice[evidence['Li']]
    #         evidence_['Lj'] = name_nice[evidence['Lj']]
    #         evidence_['Sij'] = score_nice[evidence['Sij']]
    #         evidence_['Sjk'] = score_nice[evidence['Sjk']]
    #         nice2_prob = ut.odict(zip(name_nice, probs.tolist()))
    #         ut.print_python_code('P(Lk | {evidence}) = {cpt}'.format(
    #             evidence=(ut.repr2(evidence_, explicit=True, nobraces=True, strvals=True)),
    #             cpt=ut.repr3(nice2_prob, precision=3, align=True, key_order_metric='-val')
    #         ))

    # for case in special_cases:
    #     test_data = case.drop('Lk', axis=1)
    #     test_data = test_data.drop('Lj', axis=1)
    #     test_data = test_data.reset_index(drop=True)
    #     print('----')
    #     for i in range(test_data.shape[0]):
    #         evidence = test_data.loc[i].to_dict()
    #         query_vars = ['Lk', 'Lj']
    #         probs = name_belief.query(query_vars, evidence)
    #         for queryvar in query_vars:
    #             factor = probs[queryvar]
    #             print(factor._str('phi'))
    #             probs = factor.values
    #             evidence_ = evidence.copy()
    #             evidence_['Li'] = name_nice[evidence['Li']]
    #             evidence_['Sij'] = score_nice[evidence['Sij']]
    #             evidence_['Sjk'] = score_nice[evidence['Sjk']]
    #             nice2_prob = ut.odict(zip([queryvar + '=' + x for x in name_nice], probs.tolist()))
    #             ut.print_python_code('P({queryvar} | {evidence}) = {cpt}'.format(
    #                 query_var=query_var,
    #                 evidence=(ut.repr2(evidence_, explicit=True, nobraces=True, strvals=True)),
    #                 cpt=ut.repr3(nice2_prob, precision=3, align=True, key_order_metric='-val')
    #             ))

    # _ draw model

    import plottool as pt
    import networkx as netx
    fig = pt.figure()  # NOQA
    fig.clf()
    ax = pt.gca()

    netx_nodes = [(node, {}) for node in name_model.nodes()]
    netx_edges = [(etup[0], etup[1], {}) for etup in name_model.edges()]
    netx_graph = netx.DiGraph()
    netx_graph.add_nodes_from(netx_nodes)
    netx_graph.add_edges_from(netx_edges)

    # pos = netx.graphviz_layout(netx_graph)
    pos = netx.pydot_layout(netx_graph, prog='dot')
    netx.draw(netx_graph, pos=pos, ax=ax, with_labels=True)

    pt.plt.savefig('foo.png')
    ut.startfile('foo.png')


def bayesnet_examples():
    from pgmpy.factors import TabularCPD
    from pgmpy.models import BayesianModel
    import pandas as pd

    student_model = BayesianModel([('D', 'G'),
                                   ('I', 'G'),
                                   ('G', 'L'),
                                   ('I', 'S')])
    # we can generate some random data.
    raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
    data = pd.DataFrame(raw_data, columns=['D', 'I', 'G', 'L', 'S'])
    data_train = data[: int(data.shape[0] * 0.75)]
    student_model.fit(data_train)
    student_model.get_cpds()

    data_test = data[int(0.75 * data.shape[0]): data.shape[0]]
    data_test.drop('D', axis=1, inplace=True)
    student_model.predict(data_test)

    grade_cpd = TabularCPD(
        variable='G',
        variable_card=3,
        values=[[0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7, 0.02, 0.2]],
        evidence=['I', 'D'],
        evidence_card=[2, 2])
    difficulty_cpd = TabularCPD(
        variable='D',
        variable_card=2,
        values=[[0.6, 0.4]])
    intel_cpd = TabularCPD(
        variable='I',
        variable_card=2,
        values=[[0.7, 0.3]])
    letter_cpd = TabularCPD(
        variable='L',
        variable_card=2,
        values=[[0.1, 0.4, 0.99],
                [0.9, 0.6, 0.01]],
        evidence=['G'],
        evidence_card=[3])
    sat_cpd = TabularCPD(
        variable='S',
        variable_card=2,
        values=[[0.95, 0.2],
                [0.05, 0.8]],
        evidence=['I'],
        evidence_card=[2])
    student_model.add_cpds(grade_cpd, difficulty_cpd,
                           intel_cpd, letter_cpd,
                           sat_cpd)
