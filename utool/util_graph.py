# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
try:
    import numpy as np
except ImportError:
    pass
try:
    import networkx as nx
except ImportError:
    pass
import collections
import functools
from utool import util_inject
from utool import util_const
from six.moves import reduce, zip, range
import itertools as it
(print, rrr, profile) = util_inject.inject2(__name__)


def nx_topsort_nodes(graph, nodes):
    import utool as ut
    node_rank = ut.nx_topsort_rank(graph, nodes)
    node_idx = ut.rebase_labels(node_rank)
    sorted_nodes = ut.take(nodes, node_idx)
    return sorted_nodes


def nx_topsort_rank(graph, nodes=None):
    """
    graph = inputs.exi_graph.reverse()
    nodes = flat_node_order_
    """
    import utool as ut
    if False:
        # Determenistic version
        # Ok, this doesn't work.
        dag_ranks = nx_dag_node_rank(graph, nodes)
        toprank = ut.argsort(dag_ranks, list(map(str, nodes)))
    else:
        # Non-determenistic version
        dag_ranks = nx_dag_node_rank(graph, nodes)
        topsort = list(nx.topological_sort(graph))
        # print('topsort = %r' % (topsort,))
        node_to_top_rank = ut.make_index_lookup(topsort)
        toprank = ut.dict_take(node_to_top_rank, nodes)
    return toprank


def nx_common_descendants(graph, node1, node2):
    descendants1 = nx.descendants(graph, node1)
    descendants2 = nx.descendants(graph, node2)
    common_descendants = set.intersection(descendants1, descendants2)
    return common_descendants


def nx_common_ancestors(graph, node1, node2):
    ancestors1 = nx.ancestors(graph, node1)
    ancestors2 = nx.ancestors(graph, node2)
    common_ancestors = set.intersection(ancestors1, ancestors2)
    return common_ancestors


def nx_make_adj_matrix(G):
    import utool as ut
    nodes = list(G.nodes())
    node2_idx = ut.make_index_lookup(nodes)
    edges = list(G.edges())
    edge2_idx = ut.partial(ut.dict_take, node2_idx)
    uv_list = ut.lmap(edge2_idx, edges)
    A = np.zeros((len(nodes), len(nodes)))
    A[tuple(np.array(uv_list).T)] = 1
    return A


def nx_transitive_reduction(G, mode=1):
    r"""
    References:
        https://en.wikipedia.org/wiki/Transitive_reduction#Computing_the_reduction_using_the_closure
        http://dept-info.labri.fr/~thibault/tmp/0201008.pdf
        http://stackoverflow.com/questions/17078696/transitive-reduction-of-directed-graph-in-python

    CommandLine:
        python -m utool.util_graph nx_transitive_reduction --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> G = nx.DiGraph([('a', 'b'), ('a', 'c'), ('a', 'e'),
        >>>                 ('a', 'd'), ('b', 'd'), ('c', 'e'),
        >>>                 ('d', 'e'), ('c', 'e'), ('c', 'd')])
        >>> G = testdata_graph()[1]
        >>> G_tr = nx_transitive_reduction(G, mode=1)
        >>> G_tr2 = nx_transitive_reduction(G, mode=1)
        >>> ut.quit_if_noshow()
        >>> try:
        >>>     import plottool_ibeis as pt
        >>> except ImportError:
        >>>     import plottool as pt
        >>> G_ = nx.dag.transitive_closure(G)
        >>> pt.show_nx(G    , pnum=(1, 5, 1), fnum=1)
        >>> pt.show_nx(G_tr , pnum=(1, 5, 2), fnum=1)
        >>> pt.show_nx(G_tr2 , pnum=(1, 5, 3), fnum=1)
        >>> pt.show_nx(G_   , pnum=(1, 5, 4), fnum=1)
        >>> pt.show_nx(nx.dag.transitive_closure(G_tr), pnum=(1, 5, 5), fnum=1)
        >>> ut.show_if_requested()
    """

    import utool as ut
    has_cycles = not nx.is_directed_acyclic_graph(G)
    if has_cycles:
        # FIXME: this does not work for cycle graphs.
        # Need to do algorithm on SCCs
        G_orig = G
        G = nx.condensation(G_orig)

    nodes = list(G.nodes())
    node2_idx = ut.make_index_lookup(nodes)

    # For each node u, perform DFS consider its set of (non-self) children C.
    # For each descendant v, of a node in C, remove any edge from u to v.

    if mode == 1:
        G_tr = G.copy()

        for parent in G_tr.nodes():
            # Remove self loops
            if G_tr.has_edge(parent, parent):
                G_tr.remove_edge(parent, parent)
            # For each child of the parent
            for child in list(G_tr.successors(parent)):
                # Preorder nodes includes its argument (no added complexity)
                for gchild in list(G_tr.successors(child)):
                    # Remove all edges from parent to non-child descendants
                    for descendant in nx.dfs_preorder_nodes(G_tr, gchild):
                        if G_tr.has_edge(parent, descendant):
                            G_tr.remove_edge(parent, descendant)

        if has_cycles:
            # Uncondense graph
            uncondensed_G_tr = G.__class__()
            mapping = G.graph['mapping']
            uncondensed_G_tr.add_nodes_from(mapping.keys())
            inv_mapping = ut.invert_dict(mapping, unique_vals=False)
            for u, v in G_tr.edges():
                u_ = inv_mapping[u][0]
                v_ = inv_mapping[v][0]
                uncondensed_G_tr.add_edge(u_, v_)

            for key, path in inv_mapping.items():
                if len(path) > 1:
                    directed_cycle = list(ut.itertwo(path, wrap=True))
                    uncondensed_G_tr.add_edges_from(directed_cycle)
            G_tr = uncondensed_G_tr

    else:

        def make_adj_matrix(G):
            edges = list(G.edges())
            edge2_idx = ut.partial(ut.dict_take, node2_idx)
            uv_list = ut.lmap(edge2_idx, edges)
            A = np.zeros((len(nodes), len(nodes)))
            A[tuple(np.array(uv_list).T)] = 1
            return A

        G_ = nx.dag.transitive_closure(G)

        A = make_adj_matrix(G)
        B = make_adj_matrix(G_)

        #AB = A * B
        #AB = A.T.dot(B)
        AB = A.dot(B)
        #AB = A.dot(B.T)

        A_and_notAB = np.logical_and(A, np.logical_not(AB))
        tr_uvs = np.where(A_and_notAB)

        #nodes = G.nodes()
        edges = list(zip(*ut.unflat_take(nodes, tr_uvs)))

        G_tr = G.__class__()
        G_tr.add_nodes_from(nodes)
        G_tr.add_edges_from(edges)

        if has_cycles:
            # Uncondense graph
            uncondensed_G_tr = G.__class__()
            mapping = G.graph['mapping']
            uncondensed_G_tr.add_nodes_from(mapping.keys())
            inv_mapping = ut.invert_dict(mapping, unique_vals=False)
            for u, v in G_tr.edges():
                u_ = inv_mapping[u][0]
                v_ = inv_mapping[v][0]
                uncondensed_G_tr.add_edge(u_, v_)

            for key, path in inv_mapping.items():
                if len(path) > 1:
                    directed_cycle = list(ut.itertwo(path, wrap=True))
                    uncondensed_G_tr.add_edges_from(directed_cycle)
            G_tr = uncondensed_G_tr
    return G_tr


def nx_source_nodes(graph):
    # for node in nx.dag.topological_sort(graph):
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            yield node


def nx_sink_nodes(graph):
    # for node in nx.dag.topological_sort(graph):
    for node in graph.nodes():
        if graph.out_degree(node) == 0:
            yield node


# def nx_sink_nodes(graph):
#     topsort_iter = nx.dag.topological_sort(graph)
#     sink_iter = (node for node in topsort_iter
#                  if graph.out_degree(node) == 0)
#     return sink_iter


def nx_to_adj_dict(graph):
    import utool as ut
    adj_dict = ut.ddict(list)
    for u, edges in graph.adjacency():
        adj_dict[u].extend(list(edges.keys()))
    adj_dict = dict(adj_dict)
    return adj_dict


def nx_from_adj_dict(adj_dict, cls=None):
    if cls is None:
        cls = nx.DiGraph
    nodes = list(adj_dict.keys())
    edges = [(u, v) for u, adj in adj_dict.items() for v in adj]
    graph = cls()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def nx_dag_node_rank(graph, nodes=None):
    r"""
    Returns rank of nodes that define the "level" each node is on in a
    topological sort. This is the same as the Graphviz dot rank.

    Ignore:
        simple_graph = ut.simplify_graph(exi_graph)
        adj_dict = ut.nx_to_adj_dict(simple_graph)
        import plottool as pt
        pt.qt4ensure()
        pt.show_nx(graph)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> adj_dict = {0: [5], 1: [5], 2: [1], 3: [4], 4: [0], 5: [], 6: [4], 7: [9], 8: [6], 9: [1]}
        >>> nodes = [2, 1, 5]
        >>> f_graph = ut.nx_from_adj_dict(adj_dict, nx.DiGraph)
        >>> graph = f_graph.reverse()
        >>> #ranks = ut.nx_dag_node_rank(graph, nodes)
        >>> ranks = ut.nx_dag_node_rank(graph, nodes)
        >>> result = ('ranks = %r' % (ranks,))
        >>> print(result)
        ranks = [3, 2, 1]
    """
    import utool as ut
    source = list(ut.nx_source_nodes(graph))[0]
    longest_paths = dict([(target, dag_longest_path(graph, source, target))
                          for target in graph.nodes()])
    node_to_rank = ut.map_dict_vals(len, longest_paths)
    if nodes is None:
        return node_to_rank
    else:
        ranks = ut.dict_take(node_to_rank, nodes)
        return ranks


def nx_all_nodes_between(graph, source, target, data=False):
    """
    Find all nodes with on paths between source and target.
    """
    import utool as ut
    if source is None:
        # assume there is a single source
        sources = list(ut.nx_source_nodes(graph))
        assert len(sources) == 1, (
            'specify source if there is not only one')
        source = sources[0]
    if target is None:
        # assume there is a single source
        sinks = list(ut.nx_sink_nodes(graph))
        assert len(sinks) == 1, (
            'specify sink if there is not only one')
        target = sinks[0]
    all_simple_paths = list(nx.all_simple_paths(graph, source, target))
    nodes = sorted(set.union(*map(set, all_simple_paths)))
    return nodes


def nx_all_simple_edge_paths(G, source, target, cutoff=None, keys=False,
                             data=False):
    """
    Returns each path from source to target as a list of edges.

    This function is meant to be used with MultiGraphs or MultiDiGraphs.
    When ``keys`` is True each edge in the path is returned with its unique key
    identifier. In this case it is possible to distinguish between different
    paths along different edges between the same two nodes.

    Derived from simple_paths.py in networkx
    """
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return
    import utool as ut
    import six
    visited_nodes = [source]
    visited_edges = []
    if G.is_multigraph():
        get_neighbs = ut.partial(G.edges, keys=keys, data=data)
    else:
        get_neighbs = ut.partial(G.edges, data=data)
    edge_stack = [iter(get_neighbs(source))]
    while edge_stack:
        children_edges = edge_stack[-1]
        child_edge = six.next(children_edges, None)
        if child_edge is None:
            edge_stack.pop()
            visited_nodes.pop()
            if len(visited_edges) > 0:
                visited_edges.pop()
        elif len(visited_nodes) < cutoff:
            child_node = child_edge[1]
            if child_node == target:
                yield visited_edges + [child_edge]
            elif child_node not in visited_nodes:
                visited_nodes.append(child_node)
                visited_edges.append(child_edge)
                edge_stack.append(iter(get_neighbs(child_node)))
        else:
            for edge in [child_edge] + list(children_edges):
                if edge[1] == target:
                    yield visited_edges + [edge]
            edge_stack.pop()
            visited_nodes.pop()
            if len(visited_edges) > 0:
                visited_edges.pop()


def nx_edges_between(graph, nodes1, nodes2=None, assume_disjoint=False,
                     assume_sparse=True):
    r"""
    Get edges between two components or within a single component

    Args:
        graph (nx.Graph): the graph
        nodes1 (set): list of nodes
        nodes2 (set): (default=None) if None it is equivlanet to nodes2=nodes1
        assume_disjoint (bool): skips expensive check to ensure edges arnt
            returned twice (default=False)

    CommandLine:
        python -m utool.util_graph --test-nx_edges_between

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> edges = [
        >>>     (1, 2), (2, 3), (3, 4), (4, 1), (4, 3),  # cc 1234
        >>>     (1, 5), (7, 2), (5, 1),  # cc 567 / 5678
        >>>     (7, 5), (5, 6), (8, 7),
        >>> ]
        >>> digraph = nx.DiGraph(edges)
        >>> graph = nx.Graph(edges)
        >>> nodes1 = [1, 2, 3, 4]
        >>> nodes2 = [5, 6, 7]
        >>> n2 = sorted(nx_edges_between(graph, nodes1, nodes2))
        >>> n4 = sorted(nx_edges_between(graph, nodes1))
        >>> n5 = sorted(nx_edges_between(graph, nodes1, nodes1))
        >>> n1 = sorted(nx_edges_between(digraph, nodes1, nodes2))
        >>> n3 = sorted(nx_edges_between(digraph, nodes1))
        >>> print('n2 == %r' % (n2,))
        >>> print('n4 == %r' % (n4,))
        >>> print('n5 == %r' % (n5,))
        >>> print('n1 == %r' % (n1,))
        >>> print('n3 == %r' % (n3,))
        >>> assert n2 == ([(1, 5), (2, 7)]), '2'
        >>> assert n4 == ([(1, 2), (1, 4), (2, 3), (3, 4)]), '4'
        >>> assert n5 == ([(1, 2), (1, 4), (2, 3), (3, 4)]), '5'
        >>> assert n1 == ([(1, 5), (5, 1), (7, 2)]), '1'
        >>> assert n3 == ([(1, 2), (2, 3), (3, 4), (4, 1), (4, 3)]), '3'
        >>> n6 = sorted(nx_edges_between(digraph, nodes1 + [6], nodes2 + [1, 2], assume_sparse=True))
        >>> print('n6 = %r' % (n6,))
        >>> n6 = sorted(nx_edges_between(digraph, nodes1 + [6], nodes2 + [1, 2], assume_sparse=False))
        >>> print('n6 = %r' % (n6,))
        >>> assert n6 == ([(1, 2), (1, 5), (2, 3), (4, 1), (5, 1), (5, 6), (7, 2)]), '6'

    Timeit:
        from utool.util_graph import *  # NOQA
        # ut.timeit_compare()
        import networkx as nx
        import utool as ut
        graph = nx.fast_gnp_random_graph(1000, .001)
        list(nx.connected_components(graph))
        rng = np.random.RandomState(0)
        nodes1 = set(rng.choice(list(graph.nodes()), 500, replace=False))
        nodes2 = set(graph.nodes()) - nodes1
        edges_between = ut.nx_edges_between
        %timeit list(edges_between(graph, nodes1, nodes2, assume_sparse=False, assume_disjoint=True))
        %timeit list(edges_between(graph, nodes1, nodes2, assume_sparse=False, assume_disjoint=False))
        %timeit list(edges_between(graph, nodes1, nodes2, assume_sparse=True, assume_disjoint=False))
        %timeit list(edges_between(graph, nodes1, nodes2, assume_sparse=True, assume_disjoint=True))

        graph = nx.fast_gnp_random_graph(1000, .1)
        rng = np.random.RandomState(0)
        print(graph.number_of_edges())
        nodes1 = set(rng.choice(list(graph.nodes()), 500, replace=False))
        nodes2 = set(graph.nodes()) - nodes1
        edges_between = ut.nx_edges_between
        %timeit list(edges_between(graph, nodes1, nodes2, assume_sparse=True, assume_disjoint=True))
        %timeit list(edges_between(graph, nodes1, nodes2, assume_sparse=False, assume_disjoint=True))

    Ignore:
        graph = nx.DiGraph(edges)
        graph = nx.Graph(edges)
        nodes1 = [1, 2, 3, 4]
        nodes2 = nodes1

    """
    if assume_sparse:
        # Method 1 is where we check the intersection of existing edges
        # and the edges in the second set (faster for sparse graphs)

        # helpers nx_edges between
        def _node_combo_lower(graph, both):
            both_lower = set([])
            for u in both:
                neighbs = set(graph.adj[u])
                neighbsBB_lower = neighbs.intersection(both_lower)
                for v in neighbsBB_lower:
                    yield (u, v)
                both_lower.add(u)

        def _node_combo_upper(graph, both):
            both_upper = both.copy()
            for u in both:
                neighbs = set(graph.adj[u])
                neighbsBB_upper = neighbs.intersection(both_upper)
                for v in neighbsBB_upper:
                    yield (u, v)
                both_upper.remove(u)

        def _node_product(graph, only1, only2):
            for u in only1:
                neighbs = set(graph.adj[u])
                neighbs12 = neighbs.intersection(only2)
                for v in neighbs12:
                    yield (u, v)

        # Test for special cases
        if nodes2 is None or nodes2 is nodes1:
            # Case where we just are finding internal edges
            both = set(nodes1)
            if graph.is_directed():
                edge_sets = (
                    _node_combo_upper(graph, both),  # B-to-B (upper)
                    _node_combo_lower(graph, both),  # B-to-B (lower)
                )
            else:
                edge_sets = (
                    _node_combo_upper(graph, both),  # B-to-B (upper)
                )
        elif assume_disjoint:
            # Case where we find edges between disjoint sets
            only1 = set(nodes1)
            only2 = set(nodes2)
            if graph.is_directed():
                edge_sets = (
                    _node_product(graph, only1, only2),  # 1-to-2
                    _node_product(graph, only2, only1),  # 2-to-1
                )
            else:
                edge_sets = (
                    _node_product(graph, only1, only2),  # 1-to-2
                )
        else:
            # Full general case
            nodes1_ = set(nodes1)
            if nodes2 is None:
                nodes2_ = nodes1_
            else:
                nodes2_ = set(nodes2)
            both = nodes1_.intersection(nodes2_)
            only1 = nodes1_ - both
            only2 = nodes2_ - both

            # This could be made faster by avoiding duplicate
            # calls to set(graph.adj[u]) in the helper functions
            if graph.is_directed():
                edge_sets = (
                    _node_product(graph, only1, only2),  # 1-to-2
                    _node_product(graph, only1, both),   # 1-to-B
                    _node_combo_upper(graph, both),      # B-to-B (u)
                    _node_combo_lower(graph, both),      # B-to-B (l)
                    _node_product(graph, both, only1),   # B-to-1
                    _node_product(graph, both, only2),   # B-to-2
                    _node_product(graph, only2, both),   # 2-to-B
                    _node_product(graph, only2, only1),  # 2-to-1
                )
            else:
                edge_sets = (
                    _node_product(graph, only1, only2),  # 1-to-2
                    _node_product(graph, only1, both),   # 1-to-B
                    _node_combo_upper(graph, both),      # B-to-B (u)
                    _node_product(graph, only2, both),   # 2-to-B
                )

        for u, v in it.chain.from_iterable(edge_sets):
            yield u, v

    else:
        # Method 2 is where we enumerate all possible edges and just take the
        # ones that exist (faster for very dense graphs)
        if nodes2 is None or nodes2 is nodes1:
            edge_iter = it.combinations(nodes1, 2)
        else:
            if assume_disjoint:
                # We assume len(isect(nodes1, nodes2)) == 0
                edge_iter = it.product(nodes1, nodes2)
            else:
                # make sure a single edge is not returned twice
                # in the case where len(isect(nodes1, nodes2)) > 0
                nodes1_ = set(nodes1)
                nodes2_ = set(nodes2)
                nodes_isect = nodes1_.intersection(nodes2_)
                nodes_only1 = nodes1_ - nodes_isect
                nodes_only2 = nodes2_ - nodes_isect
                edge_sets = [it.product(nodes_only1, nodes_only2),
                             it.product(nodes_only1, nodes_isect),
                             it.product(nodes_only2, nodes_isect),
                             it.combinations(nodes_isect, 2)]
                edge_iter = it.chain.from_iterable(edge_sets)

        if graph.is_directed():
            for n1, n2 in edge_iter:
                if graph.has_edge(n1, n2):
                    yield n1, n2
                if graph.has_edge(n2, n1):
                    yield n2, n1
        else:
            for n1, n2 in edge_iter:
                if graph.has_edge(n1, n2):
                    yield n1, n2


def nx_delete_node_attr(graph, name, nodes=None):
    r"""
    Removes node attributes

    Doctest:
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> G = nx.karate_club_graph()
        >>> nx.set_node_attributes(G, name='foo', values='bar')
        >>> datas = nx.get_node_attributes(G, 'club')
        >>> assert len(nx.get_node_attributes(G, 'club')) == 34
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 34
        >>> ut.nx_delete_node_attr(G, ['club', 'foo'], nodes=[1, 2])
        >>> assert len(nx.get_node_attributes(G, 'club')) == 32
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 32
        >>> ut.nx_delete_node_attr(G, ['club'])
        >>> assert len(nx.get_node_attributes(G, 'club')) == 0
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 32
    """
    if nodes is None:
        nodes = list(graph.nodes())
    removed = 0
    # names = [name] if not isinstance(name, list) else name
    node_dict = nx_node_dict(graph)

    if isinstance(name, list):
        for node in nodes:
            for name_ in name:
                try:
                    del node_dict[node][name_]
                    removed += 1
                except KeyError:
                    pass
    else:
        for node in nodes:
            try:
                del node_dict[node][name]
                removed += 1
            except KeyError:
                pass
    return removed


@profile
def nx_delete_edge_attr(graph, name, edges=None):
    r"""
    Removes an attributes from specific edges in the graph

    Doctest:
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> G = nx.karate_club_graph()
        >>> nx.set_edge_attributes(G, name='spam', values='eggs')
        >>> nx.set_edge_attributes(G, name='foo', values='bar')
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 78
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 78
        >>> ut.nx_delete_edge_attr(G, ['spam', 'foo'], edges=[(1, 2)])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 77
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 77
        >>> ut.nx_delete_edge_attr(G, ['spam'])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 0
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 77

    Doctest:
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> G = nx.MultiGraph()
        >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (4, 5), (1, 2)])
        >>> nx.set_edge_attributes(G, name='spam', values='eggs')
        >>> nx.set_edge_attributes(G, name='foo', values='bar')
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 6
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 6
        >>> ut.nx_delete_edge_attr(G, ['spam', 'foo'], edges=[(1, 2, 0)])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 5
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 5
        >>> ut.nx_delete_edge_attr(G, ['spam'])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 0
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 5
    """
    removed = 0
    keys = [name] if not isinstance(name, (list, tuple)) else name
    if edges is None:
        if graph.is_multigraph():
            edges = graph.edges(keys=True)
        else:
            edges = graph.edges()
    if graph.is_multigraph():
        for u, v, k in edges:
            for key_ in keys:
                try:
                    del graph[u][v][k][key_]
                    removed += 1
                except KeyError:
                    pass
    else:
        for u, v in edges:
            for key_ in keys:
                try:
                    del graph[u][v][key_]
                    removed += 1
                except KeyError:
                    pass
    return removed


def nx_delete_None_edge_attr(graph, edges=None):
    removed = 0
    if graph.is_multigraph():
        if edges is None:
            edges = list(graph.edges(keys=graph.is_multigraph()))
        for edge in edges:
            u, v, k = edge
            data = graph[u][v][k]
            for key in list(data.keys()):
                try:
                    if data[key] is None:
                        del data[key]
                        removed += 1
                except KeyError:
                    pass
    else:
        if edges is None:
            edges = list(graph.edges())
        for edge in graph.edges():
            u, v = edge
            data = graph[u][v]
            for key in list(data.keys()):
                try:
                    if data[key] is None:
                        del data[key]
                        removed += 1
                except KeyError:
                    pass
    return removed


def nx_delete_None_node_attr(graph, nodes=None):
    removed = 0
    if nodes is None:
        nodes = list(graph.nodes())
    for node in graph.nodes():
        node_dict = nx_node_dict(graph)
        data = node_dict[node]
        for key in list(data.keys()):
            try:
                if data[key] is None:
                    del data[key]
                    removed += 1
            except KeyError:
                pass
    return removed


def nx_set_default_node_attributes(graph, key, val):
    unset_nodes = [n for n, d in graph.nodes(data=True) if key not in d]
    if isinstance(val, dict):
        values = {n: val[n] for n in unset_nodes if n in val}
    else:
        values = {n: val for n in unset_nodes}
    nx.set_node_attributes(graph, name=key, values=values)


def nx_set_default_edge_attributes(graph, key, val):
    unset_edges = [(u, v) for u, v, d in graph.edges(data=True) if key not in d]
    if isinstance(val, dict):
        values = {e: val[e] for e in unset_edges if e in val}
    else:
        values = {e: val for e in unset_edges}
    nx.set_edge_attributes(graph, name=key, values=values)


def nx_get_default_edge_attributes(graph, key, default=None):
    import utool as ut
    edge_list = list(graph.edges())
    partial_attr_dict = nx.get_edge_attributes(graph, key)
    attr_dict = ut.dict_subset(partial_attr_dict, edge_list, default=default)
    return attr_dict


def nx_get_default_node_attributes(graph, key, default=None):
    import utool as ut
    node_list = list(graph.nodes())
    partial_attr_dict = nx.get_node_attributes(graph, key)
    attr_dict = ut.dict_subset(partial_attr_dict, node_list, default=default)
    return attr_dict


def nx_gen_node_values(G, key, nodes, default=util_const.NoParam):
    """
    Generates attributes values of specific nodes
    """
    node_dict = nx_node_dict(G)
    if default is util_const.NoParam:
        return (node_dict[n][key] for n in nodes)
    else:
        return (node_dict[n].get(key, default) for n in nodes)


def nx_gen_node_attrs(G, key, nodes=None, default=util_const.NoParam,
                      on_missing='error', on_keyerr='default'):
    r"""
    Improved generator version of nx.get_node_attributes

    Args:
        on_missing (str): Strategy for handling nodes missing from G.
            Can be {'error', 'default', 'filter'}.  defaults to 'error'.
        on_keyerr (str): Strategy for handling keys missing from node dicts.
            Can be {'error', 'default', 'filter'}.  defaults to 'default'
            if default is specified, otherwise defaults to 'error'.

    Notes:
        strategies are:
            error - raises an error if key or node does not exist
            default - returns node, but uses value specified by default
            filter - skips the node

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> G = nx.Graph([(1, 2), (2, 3)])
        >>> nx.set_node_attributes(G, name='part', values={1: 'bar', 3: 'baz'})
        >>> nodes = [1, 2, 3, 4]
        >>> #
        >>> assert len(list(ut.nx_gen_node_attrs(G, 'part', default=None, on_missing='error', on_keyerr='default'))) == 3
        >>> assert len(list(ut.nx_gen_node_attrs(G, 'part', default=None, on_missing='error', on_keyerr='filter'))) == 2
        >>> ut.assert_raises(KeyError, list, ut.nx_gen_node_attrs(G, 'part', on_missing='error', on_keyerr='error'))
        >>> #
        >>> assert len(list(ut.nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='filter', on_keyerr='default'))) == 3
        >>> assert len(list(ut.nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='filter', on_keyerr='filter'))) == 2
        >>> ut.assert_raises(KeyError, list, ut.nx_gen_node_attrs(G, 'part', nodes, on_missing='filter', on_keyerr='error'))
        >>> #
        >>> assert len(list(ut.nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='default', on_keyerr='default'))) == 4
        >>> assert len(list(ut.nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='default', on_keyerr='filter'))) == 2
        >>> ut.assert_raises(KeyError, list, ut.nx_gen_node_attrs(G, 'part', nodes, on_missing='default', on_keyerr='error'))

    Example:
        >>> # DISABLE_DOCTEST
        >>> # ALL CASES
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> G = nx.Graph([(1, 2), (2, 3)])
        >>> nx.set_node_attributes(G, name='full', values={1: 'A', 2: 'B', 3: 'C'})
        >>> nx.set_node_attributes(G, name='part', values={1: 'bar', 3: 'baz'})
        >>> nodes = [1, 2, 3, 4]
        >>> attrs = dict(ut.nx_gen_node_attrs(G, 'full'))
        >>> input_grid = {
        >>>     'nodes': [None, (1, 2, 3, 4)],
        >>>     'key': ['part', 'full'],
        >>>     'default': [util_const.NoParam, None],
        >>> }
        >>> inputs = ut.all_dict_combinations(input_grid)
        >>> kw_grid = {
        >>>     'on_missing': ['error', 'default', 'filter'],
        >>>     'on_keyerr': ['error', 'default', 'filter'],
        >>> }
        >>> kws = ut.all_dict_combinations(kw_grid)
        >>> for in_ in inputs:
        >>>     for kw in kws:
        >>>         kw2 = ut.dict_union(kw, in_)
        >>>         #print(kw2)
        >>>         on_missing = kw['on_missing']
        >>>         on_keyerr = kw['on_keyerr']
        >>>         if on_keyerr == 'default' and in_['default'] is util_const.NoParam:
        >>>             on_keyerr = 'error'
        >>>         will_miss = False
        >>>         will_keyerr = False
        >>>         if on_missing == 'error':
        >>>             if in_['key'] == 'part' and in_['nodes'] is not None:
        >>>                 will_miss = True
        >>>             if in_['key'] == 'full' and in_['nodes'] is not None:
        >>>                 will_miss = True
        >>>         if on_keyerr == 'error':
        >>>             if in_['key'] == 'part':
        >>>                 will_keyerr = True
        >>>             if on_missing == 'default':
        >>>                 if in_['key'] == 'full' and in_['nodes'] is not None:
        >>>                     will_keyerr = True
        >>>         want_error = will_miss or will_keyerr
        >>>         gen = ut.nx_gen_node_attrs(G, **kw2)
        >>>         try:
        >>>             attrs = list(gen)
        >>>         except KeyError:
        >>>             if not want_error:
        >>>                 raise AssertionError('should not have errored')
        >>>         else:
        >>>             if want_error:
        >>>                 raise AssertionError('should have errored')

    """
    if on_missing is None:
        on_missing = 'error'
    if default is util_const.NoParam and on_keyerr == 'default':
        on_keyerr = 'error'
    if nodes is None:
        nodes = G.nodes()
    # Generate `node_data` nodes and data dictionary
    node_dict = nx_node_dict(G)
    if on_missing == 'error':
        node_data = ((n, node_dict[n]) for n in nodes)
    elif on_missing == 'filter':
        node_data = ((n, node_dict[n]) for n in nodes if n in G)
    elif on_missing == 'default':
        node_data = ((n, node_dict.get(n, {})) for n in nodes)
    else:
        raise KeyError('on_missing={} must be error, filter or default'.format(
            on_missing))
    # Get `node_attrs` desired value out of dictionary
    if on_keyerr == 'error':
        node_attrs = ((n, d[key]) for n, d in node_data)
    elif on_keyerr == 'filter':
        node_attrs = ((n, d[key]) for n, d in node_data if key in d)
    elif on_keyerr == 'default':
        node_attrs = ((n, d.get(key, default)) for n, d in node_data)
    else:
        raise KeyError('on_keyerr={} must be error filter or default'.format(on_keyerr))
    return node_attrs


def nx_gen_edge_values(G, key, edges=None, default=util_const.NoParam,
                       on_missing='error', on_keyerr='default'):
    """
    Generates attributes values of specific edges

    Args:
        on_missing (str): Strategy for handling nodes missing from G.
            Can be {'error', 'default'}.  defaults to 'error'.
        on_keyerr (str): Strategy for handling keys missing from node dicts.
            Can be {'error', 'default'}.  defaults to 'default'
            if default is specified, otherwise defaults to 'error'.
    """
    if edges is None:
        edges = G.edges()
    if on_missing is None:
        on_missing = 'error'
    if on_keyerr is None:
        on_keyerr = 'default'
    if default is util_const.NoParam and on_keyerr == 'default':
        on_keyerr = 'error'
    # Generate `data_iter` edges and data dictionary
    if on_missing == 'error':
        data_iter = (G.adj[u][v] for u, v in edges)
    elif on_missing == 'default':
        data_iter = (G.adj[u][v] if G.has_edge(u, v) else {}
                     for u, v in edges)
    else:
        raise KeyError('on_missing={} must be error, filter or default'.format(
            on_missing))
    # Get `value_iter` desired value out of dictionary
    if on_keyerr == 'error':
        value_iter = (d[key] for d in data_iter)
    elif on_keyerr == 'default':
        value_iter = (d.get(key, default) for d in data_iter)
    else:
        raise KeyError('on_keyerr={} must be error or default'.format(on_keyerr))
    return value_iter
    # if default is util_const.NoParam:
    #     return (G.adj[u][v][key] for u, v in edges)
    # else:
    #     return (G.adj[u][v].get(key, default) for u, v in edges)


def nx_gen_edge_attrs(G, key, edges=None, default=util_const.NoParam,
                      on_missing='error', on_keyerr='default'):
    """
    Improved generator version of nx.get_edge_attributes

    Args:
        on_missing (str): Strategy for handling nodes missing from G.
            Can be {'error', 'default', 'filter'}.  defaults to 'error'.
            is on_missing is not error, then we allow any edge even if the
            endpoints are not in the graph.
        on_keyerr (str): Strategy for handling keys missing from node dicts.
            Can be {'error', 'default', 'filter'}.  defaults to 'default'
            if default is specified, otherwise defaults to 'error'.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        >>> nx.set_edge_attributes(G, name='part', values={(1, 2): 'bar', (2, 3): 'baz'})
        >>> edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        >>> func = ut.partial(ut.nx_gen_edge_attrs, G, 'part', default=None)
        >>> #
        >>> assert len(list(func(on_missing='error', on_keyerr='default'))) == 3
        >>> assert len(list(func(on_missing='error', on_keyerr='filter'))) == 2
        >>> ut.assert_raises(KeyError, list, func(on_missing='error', on_keyerr='error'))
        >>> #
        >>> assert len(list(func(edges, on_missing='filter', on_keyerr='default'))) == 3
        >>> assert len(list(func(edges, on_missing='filter', on_keyerr='filter'))) == 2
        >>> ut.assert_raises(KeyError, list, func(edges, on_missing='filter', on_keyerr='error'))
        >>> #
        >>> assert len(list(func(edges, on_missing='default', on_keyerr='default'))) == 4
        >>> assert len(list(func(edges, on_missing='default', on_keyerr='filter'))) == 2
        >>> ut.assert_raises(KeyError, list, func(edges, on_missing='default', on_keyerr='error'))
    """
    if on_missing is None:
        on_missing = 'error'
    if default is util_const.NoParam and on_keyerr == 'default':
        on_keyerr = 'error'

    if edges is None:
        if G.is_multigraph():
            raise NotImplementedError('')
            # uvk_iter = G.edges(keys=True)
        else:
            edges = G.edges()
    # Generate `edge_data` edges and data dictionary
    if on_missing == 'error':
        edge_data = (((u, v), G.adj[u][v]) for u, v in edges)
    elif on_missing == 'filter':
        edge_data = (((u, v), G.adj[u][v]) for u, v in edges if G.has_edge(u, v))
    elif on_missing == 'default':
        edge_data = (((u, v), G.adj[u][v])
                     if G.has_edge(u, v) else ((u, v), {})
                     for u, v in edges)
    else:
        raise KeyError('on_missing={}'.format(on_missing))
    # Get `edge_attrs` desired value out of dictionary
    if on_keyerr == 'error':
        edge_attrs = ((e, d[key]) for e, d in edge_data)
    elif on_keyerr == 'filter':
        edge_attrs = ((e, d[key]) for e, d in edge_data if key in d)
    elif on_keyerr == 'default':
        edge_attrs = ((e, d.get(key, default)) for e, d in edge_data)
    else:
        raise KeyError('on_keyerr={}'.format(on_keyerr))
    return edge_attrs

    # if edges is None:
    #     if G.is_multigraph():
    #         edges_ = G.edges(keys=True, data=True)
    #     else:
    #         edges_ = G.edges(data=True)
    #     if default is util_const.NoParam:
    #         return ((x[:-1], x[-1][key]) for x in edges_ if key in x[-1])
    #     else:
    #         return ((x[:-1], x[-1].get(key, default)) for x in edges_)

    # else:
    #     if on_missing == 'error':
    #         uv_iter = edges
    #         uvd_iter = ((u, v, G.adj[u][v]) for u, v in uv_iter)
    #     elif on_missing == 'filter':
    #         # filter edges that don't exist
    #         uv_iter = (e for e in edges if G.has_edge(*e))
    #         uvd_iter = ((u, v, G.adj[u][v]) for u, v in uv_iter)
    #     elif on_missing == 'default':
    #         # Return default data as if it existed
    #         uvd_iter = (
    #             (u, v, G.adj[u][v])
    #             if G.has_edge(u, v) else
    #             (u, v, {})
    #             for u, v in uv_iter
    #         )
    #     else:
    #         raise KeyError('on_missing={}'.format(on_missing))

    #     if default is util_const.NoParam:
    #         # return (((u, v), d[key]) for u, v, d in uvd_iter if key in d)
    #         return (((u, v), d[key]) for u, v, d in uvd_iter)
    #     else:
    #         uvd_iter = ((u, v, G.adj[u][v]) for u, v in uv_iter)
    #         return (((u, v), d.get(key, default)) for u, v, d in uvd_iter)


def nx_from_node_edge(nodes=None, edges=None):
    graph = nx.Graph()
    if nodes:
        graph.add_nodes_from(nodes)
    if edges:
        graph.add_edges_from(edges)
    return graph


def nx_minimum_weight_component(graph, weight='weight'):
    """ A minimum weight component is an MST + all negative edges """
    mwc = nx.minimum_spanning_tree(graph, weight=weight)
    # negative edges only reduce the total weight
    neg_edges = (e for e, w in nx_gen_edge_attrs(graph, weight) if w < 0)
    mwc.add_edges_from(neg_edges)
    return mwc


def nx_from_matrix(weight_matrix, nodes=None, remove_self=True):
    import utool as ut
    import numpy as np
    if nodes is None:
        nodes = list(range(len(weight_matrix)))
    weight_list = weight_matrix.ravel()
    flat_idxs_ = np.arange(weight_matrix.size)
    multi_idxs_ = np.unravel_index(flat_idxs_, weight_matrix.shape)

    # Remove 0 weight edges
    flags = np.logical_not(np.isclose(weight_list, 0))
    weight_list = ut.compress(weight_list, flags)
    multi_idxs = ut.compress(list(zip(*multi_idxs_)), flags)
    edge_list = ut.lmap(tuple, ut.unflat_take(nodes, multi_idxs))

    if remove_self:
        flags = [e1 != e2 for e1, e2 in edge_list]
        edge_list = ut.compress(edge_list, flags)
        weight_list = ut.compress(weight_list, flags)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_list)
    label_list = ['%.2f' % w for w in weight_list]
    nx.set_edge_attributes(graph, name='weight', values=dict(zip(edge_list, weight_list)))
    nx.set_edge_attributes(graph, name='label', values=dict(zip(edge_list, label_list)))
    return graph


def nx_ensure_agraph_color(graph):
    """ changes colors to hex strings on graph attrs """
    try:
        from plottool_ibeis import color_funcs
        import plottool_ibeis as pt
    except ImportError:
        from plottool import color_funcs
        import plottool as pt
    #import six
    def _fix_agraph_color(data):
        try:
            orig_color = data.get('color', None)
            alpha = data.get('alpha', None)
            color = orig_color
            if color is None and alpha is not None:
                color = [0, 0, 0]
            if color is not None:
                color = pt.ensure_nonhex_color(color)
                #if isinstance(color, np.ndarray):
                #    color = color.tolist()
                color = list(color_funcs.ensure_base255(color))
                if alpha is not None:
                    if len(color) == 3:
                        color += [int(alpha * 255)]
                    else:
                        color[3] = int(alpha * 255)
                color = tuple(color)
                if len(color) == 3:
                    data['color'] = '#%02x%02x%02x' % color
                else:
                    data['color'] = '#%02x%02x%02x%02x' % color
        except Exception as ex:
            import utool as ut
            ut.printex(ex, keys=['color', 'orig_color', 'data'])
            raise

    for node, node_data in graph.nodes(data=True):
        data = node_data
        _fix_agraph_color(data)

    for u, v, edge_data in graph.edges(data=True):
        data = edge_data
        _fix_agraph_color(data)


def nx_edges(graph, keys=False, data=False):
    if graph.is_multigraph():
        edges = graph.edges(keys=keys, data=data)
    else:
        edges = graph.edges(data=data)
        #if keys:
        #    edges = [e[0:2] + (0,) + e[:2] for e in edges]
    return edges


def dag_longest_path(graph, source, target):
    """
    Finds the longest path in a dag between two nodes
    """
    if source == target:
        return [source]
    allpaths = nx.all_simple_paths(graph, source, target)
    longest_path = []
    for ell in allpaths:
        if len(ell) > len(longest_path):
            longest_path = ell
    return longest_path


def testdata_graph():
    r"""
    Returns:
        tuple: (graph, G)

    CommandLine:
        python -m utool.util_graph --exec-testdata_graph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> (graph, G) = testdata_graph()
        >>> import plottool as pt
        >>> ut.ensureqt()
        >>> pt.show_nx(G, layout='agraph')
        >>> ut.show_if_requested()
    """
    import utool as ut
    # Define adjacency list
    graph = {
        'a': ['b'],
        'b': ['c', 'f', 'e'],
        'c': ['g', 'd'],
        'd': ['c', 'h'],
        'e': ['a', 'f'],
        'f': ['g'],
        'g': ['f'],
        'h': ['g', 'd'],
        'i': ['j'],
        'j': [],
    }
    graph = {
        'a': ['b'],
        'b': ['c'],
        'c': ['d'],
        'd': ['a', 'e'],
        'e': ['c'],
    }
    #graph = {'a': ['b'], 'b': ['c'], 'c': ['d'], 'd': ['a']}
    #graph = {'a': ['b'], 'b': ['c'], 'c': ['d'], 'd': ['e'], 'e': ['a']}
    graph = {'a': ['b'], 'b': ['c'], 'c': ['d'], 'd': ['e'], 'e': ['a'], 'f': ['c']}
    #graph = {'a': ['b'], 'b': ['c'], 'c': ['d'], 'd': ['e'], 'e': ['b']}

    graph = {'a': ['b', 'c', 'd'], 'e': ['d'], 'f': ['d', 'e'], 'b': [], 'c': [], 'd': []}  # double pair in non-scc
    graph = {'a': ['b', 'c', 'd'], 'e': ['d'], 'f': ['d', 'e'], 'b': [], 'c': [], 'd': ['e']}  # double pair in non-scc
    #graph = {'a': ['b', 'c', 'd'], 'e': ['d', 'f'], 'f': ['d', 'e'], 'b': [], 'c': [], 'd': ['e']}  # double pair in non-scc
    #graph = {'a': ['b', 'c', 'd'], 'e': ['d', 'c'], 'f': ['d', 'e'], 'b': ['e'], 'c': ['e'], 'd': ['e']}  # double pair in non-scc
    graph = {'a': ['b', 'c', 'd'], 'e': ['d', 'c'], 'f': ['d', 'e'], 'b': ['e'], 'c': ['e', 'b'], 'd': ['e']}  # double pair in non-scc
    # Extract G = (V, E)
    nodes = list(graph.keys())
    edges = ut.flatten([[(v1, v2) for v2 in v2s] for v1, v2s in graph.items()])
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if False:
        G.remove_node('e')
        del graph['e']

        for val in graph.values():
            try:
                val.remove('e')
            except ValueError:
                pass
    return graph, G


def dict_depth(dict_, accum=0):
    if not isinstance(dict_, dict):
        return accum
    return max([dict_depth(val, accum + 1)
                for key, val in dict_.items()])


def edges_to_adjacency_list(edges):
    import utool as ut
    children_, parents_ = list(zip(*edges))
    parent_to_children = ut.group_items(parents_, children_)
    #to_leafs = {tablename: path_to_leafs(tablename, parent_to_children)}
    return parent_to_children


def paths_to_root(tablename, root, child_to_parents):
    """

    CommandLine:
        python -m utool.util_graph --exec-paths_to_root:0
        python -m utool.util_graph --exec-paths_to_root:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> child_to_parents = {
        >>>     'chip': ['dummy_annot'],
        >>>     'chipmask': ['dummy_annot'],
        >>>     'descriptor': ['keypoint'],
        >>>     'fgweight': ['keypoint', 'probchip'],
        >>>     'keypoint': ['chip'],
        >>>     'notch': ['dummy_annot'],
        >>>     'probchip': ['dummy_annot'],
        >>>     'spam': ['fgweight', 'chip', 'keypoint']
        >>> }
        >>> root = 'dummy_annot'
        >>> tablename = 'fgweight'
        >>> to_root = paths_to_root(tablename, root, child_to_parents)
        >>> result = ut.repr3(to_root)
        >>> print(result)
        {
            'keypoint': {
                'chip': {
                    'dummy_annot': None,
                },
            },
            'probchip': {
                'dummy_annot': None,
            },
        }

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> root = u'annotations'
        >>> tablename = u'Notch_Tips'
        >>> child_to_parents = {
        >>>     'Block_Curvature': [
        >>>         'Trailing_Edge',
        >>>     ],
        >>>     'Has_Notch': [
        >>>         'annotations',
        >>>     ],
        >>>     'Notch_Tips': [
        >>>         'annotations',
        >>>     ],
        >>>     'Trailing_Edge': [
        >>>         'Notch_Tips',
        >>>     ],
        >>> }
        >>> to_root = paths_to_root(tablename, root, child_to_parents)
        >>> result = ut.repr3(to_root)
        >>> print(result)
    """
    if tablename == root:
        return None
    parents = child_to_parents[tablename]
    return {parent: paths_to_root(parent, root, child_to_parents)
            for parent in parents}


def get_allkeys(dict_):
    import utool as ut
    if not isinstance(dict_, dict):
        return []
    subkeys = [[key] + get_allkeys(val)
               for key, val in dict_.items()]
    return ut.unique_ordered(ut.flatten(subkeys))


def traverse_path(start, end, seen_, allkeys, mat):
    import utool as ut
    if seen_ is None:
        seen_ = set([])
    index = allkeys.index(start)
    sub_indexes = np.where(mat[index])[0]
    if len(sub_indexes) > 0:
        subkeys = ut.take(allkeys, sub_indexes)
        # subkeys_ = ut.take(allkeys, sub_indexes)
        # subkeys = [subkey for subkey in subkeys_
        #            if subkey not in seen_]
        # for sk in subkeys:
        #     seen_.add(sk)
        if len(subkeys) > 0:
            return {subkey: traverse_path(subkey, end, seen_, allkeys, mat)
                    for subkey in subkeys}
    return None


def reverse_path(dict_, root, child_to_parents):
    """
    CommandLine:
        python -m utool.util_graph --exec-reverse_path --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> child_to_parents = {
        >>>     'chip': ['dummy_annot'],
        >>>     'chipmask': ['dummy_annot'],
        >>>     'descriptor': ['keypoint'],
        >>>     'fgweight': ['keypoint', 'probchip'],
        >>>     'keypoint': ['chip'],
        >>>     'notch': ['dummy_annot'],
        >>>     'probchip': ['dummy_annot'],
        >>>     'spam': ['fgweight', 'chip', 'keypoint']
        >>> }
        >>> to_root = {
        >>>     'fgweight': {
        >>>         'keypoint': {
        >>>             'chip': {
        >>>                 'dummy_annot': None,
        >>>             },
        >>>         },
        >>>         'probchip': {
        >>>             'dummy_annot': None,
        >>>         },
        >>>     },
        >>> }
        >>> reversed_ = reverse_path(to_root, 'dummy_annot', child_to_parents)
        >>> result = ut.repr3(reversed_)
        >>> print(result)
        {
            'dummy_annot': {
                'chip': {
                    'keypoint': {
                        'fgweight': None,
                    },
                },
                'probchip': {
                    'fgweight': None,
                },
            },
        }

    """
    # Hacky but illustrative
    # TODO; implement non-hacky version
    allkeys = get_allkeys(dict_)
    mat = np.zeros((len(allkeys), len(allkeys)))
    for key in allkeys:
        if key != root:
            for parent in child_to_parents[key]:
                rx = allkeys.index(parent)
                cx = allkeys.index(key)
                mat[rx][cx] = 1
    end = None
    seen_ = set([])
    reversed_ = {root: traverse_path(root, end, seen_, allkeys, mat)}
    return reversed_


def get_levels(dict_, n=0, levels=None):
    r"""
    DEPCIRATE

    Args:
        dict_ (dict_):  a dictionary
        n (int): (default = 0)
        levels (None): (default = None)

    CommandLine:
        python -m utool.util_graph --test-get_levels --show
        python3 -m utool.util_graph --test-get_levels --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> from_root = {
        >>>     'dummy_annot': {
        >>>         'chip': {
        >>>                 'keypoint': {
        >>>                             'fgweight': None,
        >>>                         },
        >>>             },
        >>>         'probchip': {
        >>>                 'fgweight': None,
        >>>             },
        >>>     },
        >>> }
        >>> dict_ = from_root
        >>> n = 0
        >>> levels = None
        >>> levels_ = get_levels(dict_, n, levels)
        >>> result = ut.repr2(levels_, nl=1)
        >>> print(result)
        [
            ['dummy_annot'],
            ['chip', 'probchip'],
            ['keypoint', 'fgweight'],
            ['fgweight'],
        ]
    """
    if levels is None:
        levels_ = [[] for _ in range(dict_depth(dict_))]
    else:
        levels_ = levels
    if dict_ is None:
        return []
    for key in dict_.keys():
        levels_[n].append(key)
    for val in dict_.values():
        get_levels(val, n + 1, levels_)
    return levels_


def longest_levels(levels_):
    r"""
    Args:
        levels_ (list):

    CommandLine:
        python -m utool.util_graph --exec-longest_levels --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> levels_ = [
        >>>     ['dummy_annot'],
        >>>     ['chip', 'probchip'],
        >>>     ['keypoint', 'fgweight'],
        >>>     ['fgweight'],
        >>> ]
        >>> new_levels = longest_levels(levels_)
        >>> result = ('new_levels = %s' % (ut.repr2(new_levels, nl=1),))
        >>> print(result)
        new_levels = [
            ['dummy_annot'],
            ['chip', 'probchip'],
            ['keypoint'],
            ['fgweight'],
        ]
    """
    return shortest_levels(levels_[::-1])[::-1]
    # seen_ = set([])
    # new_levels = []
    # for level in levels_[::-1]:
    #     new_level = [item for item in level if item not in seen_]
    #     seen_ = seen_.union(set(new_level))
    #     new_levels.append(new_level)
    # new_levels = new_levels[::-1]
    # return new_levels


def shortest_levels(levels_):
    r"""
    Args:
        levels_ (list):

    CommandLine:
        python -m utool.util_graph --exec-shortest_levels --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> levels_ = [
        >>>     ['dummy_annot'],
        >>>     ['chip', 'probchip'],
        >>>     ['keypoint', 'fgweight'],
        >>>     ['fgweight'],
        >>> ]
        >>> new_levels = shortest_levels(levels_)
        >>> result = ('new_levels = %s' % (ut.repr2(new_levels, nl=1),))
        >>> print(result)
        new_levels = [
            ['dummy_annot'],
            ['chip', 'probchip'],
            ['keypoint', 'fgweight'],
        ]
    """
    seen_ = set([])
    new_levels = []
    for level in levels_:
        new_level = [item for item in level if item not in seen_]
        seen_ = seen_.union(set(new_level))
        if len(new_level) > 0:
            new_levels.append(new_level)
    new_levels = new_levels
    return new_levels


def simplify_graph(graph):
    """
    strips out everything but connectivity

    Args:
        graph (nx.Graph):

    Returns:
        nx.Graph: new_graph

    CommandLine:
        python3 -m utool.util_graph simplify_graph --show
        python2 -m utool.util_graph simplify_graph --show

        python2 -c "import networkx as nx; print(nx.__version__)"
        python3 -c "import networkx as nx; print(nx.__version__)"

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> graph = nx.DiGraph([('a', 'b'), ('a', 'c'), ('a', 'e'),
        >>>                     ('a', 'd'), ('b', 'd'), ('c', 'e'),
        >>>                     ('d', 'e'), ('c', 'e'), ('c', 'd')])
        >>> new_graph = simplify_graph(graph)
        >>> result = ut.repr2(list(new_graph.edges()))
        >>> #adj_list = sorted(list(nx.generate_adjlist(new_graph)))
        >>> #result = ut.repr2(adj_list)
        >>> print(result)
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (2, 3), (2, 4), (3, 4)]

        ['0 1 2 3 4', '1 3 4', '2 4', '3', '4 3']
    """
    import utool as ut
    nodes = sorted(list(graph.nodes()))
    node_lookup = ut.make_index_lookup(nodes)
    if graph.is_multigraph():
        edges = list(graph.edges(keys=True))
    else:
        edges = list(graph.edges())
    new_nodes = ut.take(node_lookup, nodes)
    if graph.is_multigraph():
        new_edges = [(node_lookup[e[0]], node_lookup[e[1]], e[2], {}) for e in edges]
    else:
        new_edges = [(node_lookup[e[0]], node_lookup[e[1]]) for e in edges]
    cls = graph.__class__
    new_graph = cls()
    new_graph.add_nodes_from(new_nodes)
    new_graph.add_edges_from(new_edges)
    return new_graph


def subgraph_from_edges(G, edge_list, ref_back=True):
    """
    Creates a networkx graph that is a subgraph of G
    defined by the list of edges in edge_list.

    Requires G to be a networkx MultiGraph or MultiDiGraph
    edge_list is a list of edges in either (u,v) or (u,v,d) form
    where u and v are nodes comprising an edge,
    and d would be a dictionary of edge attributes

    ref_back determines whether the created subgraph refers to back
    to the original graph and therefore changes to the subgraph's
    attributes also affect the original graph, or if it is to create a
    new copy of the original graph.

    References:
        http://stackoverflow.com/questions/16150557/nx-subgraph-from-edges
    """

    # TODO: support multi-di-graph
    sub_nodes = list({y for x in edge_list for y in x[0:2]})
    #edge_list_no_data = [edge[0:2] for edge in edge_list]
    multi_edge_list = [edge[0:3] for edge in edge_list]

    if ref_back:
        G_sub = G.subgraph(sub_nodes)
        for edge in G_sub.edges(keys=True):
            if edge not in multi_edge_list:
                G_sub.remove_edge(*edge)
    else:
        G_sub = G.subgraph(sub_nodes).copy()
        for edge in G_sub.edges(keys=True):
            if edge not in multi_edge_list:
                G_sub.remove_edge(*edge)

    return G_sub


def nx_node_dict(G):
    if nx.__version__.startswith('1'):
        return getattr(G, 'node')
    else:
        return G.nodes


def all_multi_paths(graph, source, target, data=False):
    r"""
    Returns specific paths along multi-edges from the source to this table.
    Multipaths are identified by edge keys.

    Returns all paths from source to target. This function treats multi-edges
    as distinct and returns the key value in each edge tuple that defines a
    path.

    Example:
        >>> # DISABLE_DOCTEST
        >>> from dtool.depcache_control import *  # NOQA
        >>> from utool.util_graph import *  # NOQA
        >>> from dtool.example_depcache import testdata_depc
        >>> depc = testdata_depc()
        >>> graph = depc.graph
        >>> source = depc.root
        >>> target = 'notchpair'
        >>> path_list1 = ut.all_multi_paths(graph, depc.root, 'notchpair')
        >>> path_list2 = ut.all_multi_paths(graph, depc.root, 'spam')
        >>> result1 = ('path_list1 = %s' % ut.repr3(path_list1, nl=1))
        >>> result2 = ('path_list2 = %s' % ut.repr3(path_list2, nl=2))
        >>> result = '\n'.join([result1, result2])
        >>> print(result)
        path_list1 = [
            [('dummy_annot', 'notch', 0), ('notch', 'notchpair', 0)],
            [('dummy_annot', 'notch', 0), ('notch', 'notchpair', 1)],
        ]
        path_list2 = [
            [
                ('dummy_annot', 'chip', 0),
                ('chip', 'keypoint', 0),
                ('keypoint', 'fgweight', 0),
                ('fgweight', 'spam', 0),
            ],
            [
                ('dummy_annot', 'chip', 0),
                ('chip', 'keypoint', 0),
                ('keypoint', 'spam', 0),
            ],
            [
                ('dummy_annot', 'chip', 0),
                ('chip', 'spam', 0),
            ],
            [
                ('dummy_annot', 'probchip', 0),
                ('probchip', 'fgweight', 0),
                ('fgweight', 'spam', 0),
            ],
        ]
    """
    path_multiedges = list(nx_all_simple_edge_paths(graph, source, target,
                                                    keys=True, data=data))
    return path_multiedges


def reverse_path_edges(edge_list):
    return [(edge[1], edge[0],) + tuple(edge[2:]) for edge in edge_list][::-1]


def bfs_multi_edges(G, source, reverse=False, keys=True, data=False):
    """Produce edges in a breadth-first-search starting at source.
    -----
    Based on http://www.ics.uci.edu/~eppstein/PADS/BFS.py
    by D. Eppstein, July 2004.
    """
    from collections import deque
    from functools import partial
    if reverse:
        G = G.reverse()
    edges_iter = partial(G.edges_iter, keys=keys, data=data)

    list(G.edges_iter('multitest', keys=True, data=True))

    visited_nodes = set([source])
    # visited_edges = set([])
    queue = deque([(source, edges_iter(source))])
    while queue:
        parent, edges = queue[0]
        try:
            edge = next(edges)
            edge_nodata = edge[0:3]
            # if edge_nodata not in visited_edges:
            yield edge
            # visited_edges.add(edge_nodata)
            child = edge_nodata[1]
            if child not in visited_nodes:
                visited_nodes.add(child)
                queue.append((child, edges_iter(child)))
        except StopIteration:
            queue.popleft()


def dfs_conditional(G, source, state, can_cross):
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *
        >>> G = nx.Graph()
        >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
        >>> G.adj[2][3]['lava'] = True
        >>> G.adj[3][4]['lava'] = True
        >>> def can_cross(G, edge, state):
        >>>     # can only cross lava once, then your lava protection wears off
        >>>     data = G.get_edge_data(*edge)
        >>>     lava = int(data.get('lava', False))
        >>>     if not lava or state == 0:
        >>>         return True, state + lava
        >>>     return False, lava
        >>> assert 5 not in dfs_conditional(G, 1, state=0, can_cross=can_cross)
        >>> G.adj[3][4]['lava'] = False
        >>> assert 5 in dfs_conditional(G, 1, state=0, can_cross=can_cross)
    """
    # stack based version
    visited = {source}
    stack = [(source, iter(G[source]), state)]
    while stack:
        parent, children, state = stack[-1]
        try:
            child = next(children)
            if child not in visited:
                edge = (parent, child)
                flag, new_state = can_cross(G, edge, state)
                if flag:
                    yield child
                    visited.add(child)
                    stack.append((child, iter(G[child]), new_state))
        except StopIteration:
            stack.pop()


def bfs_conditional(G, source, reverse=False, keys=True, data=False,
                    yield_nodes=True, yield_if=None,
                    continue_if=None, visited_nodes=None,
                    yield_source=False):
    """
    Produce edges in a breadth-first-search starting at source, but only return
    nodes that satisfiy a condition, and only iterate past a node if it
    satisfies a different condition.

    conditions are callables that take (G, child, edge) and return true or false

    CommandLine:
        python -m utool.util_graph bfs_conditional

    Example:
        >>> # DISABLE_DOCTEST
        >>> import networkx as nx
        >>> import utool as ut
        >>> G = nx.Graph()
        >>> G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4)])
        >>> continue_if = lambda G, child, edge: True
        >>> result = list(ut.bfs_conditional(G, 1, yield_nodes=False))
        >>> print(result)
        [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (4, 2)]

    Example:
        >>> # ENABLE_DOCTEST
        >>> import networkx as nx
        >>> import utool as ut
        >>> G = nx.Graph()
        >>> continue_if = lambda G, child, edge: (child % 2 == 0)
        >>> yield_if = lambda G, child, edge: (child % 2 == 1)
        >>> G.add_edges_from([(0, 1), (1, 3), (3, 5), (5, 10),
        >>>                   (4, 3), (3, 6),
        >>>                   (0, 2), (2, 4), (4, 6), (6, 10)])
        >>> result = list(ut.bfs_conditional(G, 0, continue_if=continue_if,
        >>>                                  yield_if=yield_if))
        >>> print(result)
        [1, 3, 5]
    """
    if reverse and hasattr(G, 'reverse'):
        G = G.reverse()
    if isinstance(G, nx.Graph):
        neighbors = functools.partial(G.edges, data=data)
    else:
        neighbors = functools.partial(G.edges, keys=keys, data=data)

    queue = collections.deque([])

    if visited_nodes is None:
        visited_nodes = set([])
    else:
        visited_nodes = set(visited_nodes)

    if source not in visited_nodes:
        if yield_nodes and yield_source:
            yield source
        visited_nodes.add(source)
        new_edges = neighbors(source)
        if isinstance(new_edges, list):
            new_edges = iter(new_edges)
        queue.append((source, new_edges))

    while queue:
        parent, edges = queue[0]
        for edge in edges:
            child = edge[1]
            if yield_nodes:
                if child not in visited_nodes:
                    if yield_if is None or yield_if(G, child, edge):
                        yield child
            else:
                if yield_if is None or yield_if(G, child, edge):
                    yield edge
            if child not in visited_nodes:
                visited_nodes.add(child)
                # Add new children to queue if the condition is satisfied
                if continue_if is None or continue_if(G, child, edge):
                    new_edges = neighbors(child)
                    if isinstance(new_edges, list):
                        new_edges = iter(new_edges)
                    queue.append((child, new_edges))
        queue.popleft()


def color_nodes(graph, labelattr='label', brightness=.878,
                outof=None, sat_adjust=None):
    """ Colors edges and nodes by nid """
    try:
        import plottool_ibeis as pt
    except ImportError:
        import plottool as pt
    import utool as ut
    node_to_lbl = nx.get_node_attributes(graph, labelattr)
    unique_lbls = sorted(set(node_to_lbl.values()))
    ncolors = len(unique_lbls)
    if outof is None:
        if (ncolors) == 1:
            unique_colors = [pt.LIGHT_BLUE]
        elif (ncolors) == 2:
            # https://matplotlib.org/examples/color/named_colors.html
            unique_colors = ['royalblue', 'orange']
            unique_colors = list(map(pt.color_funcs.ensure_base01, unique_colors))
        else:
            unique_colors = pt.distinct_colors(ncolors, brightness=brightness)
    else:
        unique_colors = pt.distinct_colors(outof, brightness=brightness)

    if sat_adjust:
        unique_colors = [
            pt.color_funcs.adjust_hsv_of_rgb(c, sat_adjust=sat_adjust)
            for c in unique_colors
        ]
    # Find edges and aids strictly between two nids
    if outof is None:
        lbl_to_color = ut.dzip(unique_lbls, unique_colors)
    else:
        gray = pt.color_funcs.ensure_base01('lightgray')
        unique_colors = [gray] + unique_colors
        offset = max(1, min(unique_lbls)) - 1
        node_to_lbl = ut.map_vals(lambda nid: max(0, nid - offset), node_to_lbl)
        lbl_to_color = ut.dzip(range(outof + 1), unique_colors)
    node_to_color = ut.map_vals(lbl_to_color, node_to_lbl)
    nx.set_node_attributes(graph, name='color', values=node_to_color)
    ut.nx_ensure_agraph_color(graph)


def graph_info(graph, ignore=None, stats=False, verbose=False):
    import utool as ut

    node_dict = nx_node_dict(graph)
    node_attrs = list(node_dict.values())
    edge_attrs = list(ut.take_column(graph.edges(data=True), 2))

    if stats:
        import utool
        with utool.embed_on_exception_context:
            import pandas as pd
            node_df = pd.DataFrame(node_attrs)
            edge_df = pd.DataFrame(edge_attrs)
            if ignore is not None:
                ut.delete_dict_keys(node_df, ignore)
                ut.delete_dict_keys(edge_df, ignore)
            # Not really histograms anymore
            try:
                node_attr_hist = node_df.describe().to_dict()
            except ValueError:
                node_attr_hist
            try:
                edge_attr_hist = edge_df.describe().to_dict()
            except ValueError:
                edge_attr_hist = {}
            key_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            node_attr_hist = ut.map_dict_vals(lambda x: ut.order_dict_by(x, key_order), node_attr_hist)
            edge_attr_hist = ut.map_dict_vals(lambda x: ut.order_dict_by(x, key_order), edge_attr_hist)
    else:
        node_attr_hist = ut.dict_hist(ut.flatten([attr.keys() for attr in node_attrs]))
        edge_attr_hist = ut.dict_hist(ut.flatten([attr.keys() for attr in edge_attrs]))
        if ignore is not None:
            ut.delete_dict_keys(edge_attr_hist, ignore)
            ut.delete_dict_keys(node_attr_hist, ignore)
    node_type_hist = ut.dict_hist(list(map(type, graph.nodes())))
    info_dict = ut.odict([
        ('directed', graph.is_directed()),
        ('multi', graph.is_multigraph()),
        ('num_nodes', len(graph)),
        ('num_edges', len(list(graph.edges()))),
        ('edge_attr_hist', ut.sort_dict(edge_attr_hist)),
        ('node_attr_hist', ut.sort_dict(node_attr_hist)),
        ('node_type_hist', ut.sort_dict(node_type_hist)),
        ('graph_attrs', graph.graph),
        ('graph_name', graph.name),
    ])
    #unique_attrs = ut.map_dict_vals(ut.unique, ut.dict_accum(*node_attrs))
    #ut.dict_isect_combine(*node_attrs))
    #[list(attrs.keys())]
    if verbose:
        print(ut.repr3(info_dict))
    return info_dict


def get_graph_bounding_box(graph):
    # import utool as ut
    try:
        import vtool_ibeis as vt
    except ImportError:
        import vtool as vt
    #nx.get_node_attrs = nx.get_node_attributes
    nodes = list(graph.nodes())
    # pos_list = nx_gen_node_values(graph, 'pos', nodes, default=(0, 0))
    # shape_list = nx_gen_node_values(graph, 'size', nodes, default=(1, 1))
    shape_list = nx_gen_node_values(graph, 'size', nodes)
    pos_list = nx_gen_node_values(graph, 'pos', nodes)

    node_extents = np.array([
        vt.extent_from_bbox(vt.bbox_from_center_wh(xy, wh))
        for xy, wh in zip(pos_list, shape_list)
    ])
    tl_x, br_x, tl_y, br_y = node_extents.T
    extent = tl_x.min(), br_x.max(), tl_y.min(), br_y.max()
    bbox = vt.bbox_from_extent(extent)
    return bbox


def translate_graph(graph, t_xy):
    #import utool as ut
    import utool as ut
    node_pos_attrs = ['pos']
    for attr in node_pos_attrs:
        attrdict = nx.get_node_attributes(graph, attr)
        attrdict = {
            node: pos + t_xy
            for node, pos in attrdict.items()
        }
        nx.set_node_attributes(graph, name=attr, values=attrdict)
    edge_pos_attrs = ['ctrl_pts', 'end_pt', 'head_lp', 'lp', 'start_pt', 'tail_lp']
    ut.nx_delete_None_edge_attr(graph)
    for attr in edge_pos_attrs:
        attrdict = nx.get_edge_attributes(graph, attr)
        attrdict = {
            node: pos + t_xy
            if pos is not None else pos
            for node, pos in attrdict.items()
        }
        nx.set_edge_attributes(graph, name=attr, values=attrdict)


def translate_graph_to_origin(graph):
    x, y, w, h = get_graph_bounding_box(graph)
    translate_graph(graph, (-x, -y))


def stack_graphs(graph_list, vert=False, pad=None):
    import utool as ut
    graph_list_ = [g.copy() for g in graph_list]
    for g in graph_list_:
        translate_graph_to_origin(g)
    bbox_list = [get_graph_bounding_box(g) for g in graph_list_]
    if vert:
        dim1 = 3
        dim2 = 2
    else:
        dim1 = 2
        dim2 = 3
    dim1_list = np.array([bbox[dim1] for bbox in bbox_list])
    dim2_list = np.array([bbox[dim2] for bbox in bbox_list])
    if pad is None:
        pad = np.mean(dim1_list) / 2
    offset1_list = ut.cumsum([0] + [d + pad for d in dim1_list[:-1]])
    max_dim2 = max(dim2_list)
    offset2_list = [(max_dim2 - d2) / 2 for d2 in dim2_list]
    if vert:
        t_xy_list = [(d2, d1) for d1, d2 in zip(offset1_list, offset2_list)]
    else:
        t_xy_list = [(d1, d2) for d1, d2 in zip(offset1_list, offset2_list)]

    for g, t_xy in zip(graph_list_, t_xy_list):
        translate_graph(g, t_xy)
        nx.set_node_attributes(g, name='pin', values='true')

    new_graph = nx.compose_all(graph_list_)
    #pt.show_nx(new_graph, layout='custom', node_labels=False, as_directed=False)  # NOQA
    return new_graph


def nx_contracted_nodes(G, u, v, self_loops=True, inplace=False):
    """
    copy of networkx function with inplace modification
    TODO: commit to networkx
    """
    import itertools as it
    if G.is_directed():
        in_edges = ((w, u, d) for w, x, d in G.in_edges(v, data=True)
                    if self_loops or w != u)
        out_edges = ((u, w, d) for x, w, d in G.out_edges(v, data=True)
                     if self_loops or w != u)
        new_edges = it.chain(in_edges, out_edges)
    else:
        new_edges = ((u, w, d) for x, w, d in G.edges(v, data=True)
                     if self_loops or w != u)
    if inplace:
        H = G
        new_edges = list(new_edges)
    else:
        H = G.copy()
    node_dict = nx_node_dict(H)
    v_data = node_dict[v]
    H.remove_node(v)
    H.add_edges_from(new_edges)
    if 'contraction' in node_dict[u]:
        node_dict[u]['contraction'][v] = v_data
    else:
        node_dict[u]['contraction'] = {v: v_data}
    return H


def approx_min_num_components(nodes, negative_edges):
    """
    Find approximate minimum number of connected components possible
    Each edge represents that two nodes must be separated

    This code doesn't solve the problem. The problem is NP-complete and
    reduces to minimum clique cover (MCC). This is only an approximate
    solution. Not sure what the approximation ratio is.

    CommandLine:
        python -m utool.util_graph approx_min_num_components

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> edges = [(1, 2), (2, 3), (3, 1),
        >>>          (4, 5), (5, 6), (6, 4),
        >>>          (7, 8), (8, 9), (9, 7),
        >>>          (1, 4), (4, 7), (7, 1),
        >>>         ]
        >>> g_pos = nx.Graph()
        >>> g_pos.add_edges_from(edges)
        >>> g_neg = nx.complement(g_pos)
        >>> #import plottool as pt
        >>> #pt.qt4ensure()
        >>> #pt.show_nx(g_pos)
        >>> #pt.show_nx(g_neg)
        >>> negative_edges = g_neg.edges()
        >>> nodes = [1, 2, 3, 4, 5, 6, 7]
        >>> negative_edges = [(1, 2), (2, 3), (4, 5)]
        >>> result = approx_min_num_components(nodes, negative_edges)
        >>> print(result)
        2
    """
    import utool as ut
    num = 0
    g_neg = nx.Graph()
    g_neg.add_nodes_from(nodes)
    g_neg.add_edges_from(negative_edges)

    # Collapse all nodes with degree 0
    if nx.__version__.startswith('2'):
        deg0_nodes = [n for n, d in g_neg.degree() if d == 0]
    else:
        try:
            deg0_nodes = [n for n, d in g_neg.degree_iter() if d == 0]
        except Exception:
            deg0_nodes = [n for n, d in g_neg.degree() if d == 0]

    for u, v in ut.itertwo(deg0_nodes):
        nx_contracted_nodes(g_neg, v, u, inplace=True)
        # g_neg = nx.contracted_nodes(g_neg, v, u, self_loops=False)

    # Initialize unused nodes to be everything
    unused = list(g_neg.nodes())
    # complement of the graph contains all possible positive edges
    g_pos = nx.complement(g_neg)

    if False:
        from networkx.algorithms.approximation import clique
        maxiset, cliques = clique.clique_removal(g_pos)
        num = len(cliques)
        return num

    # Iterate until we have used all nodes
    while len(unused) > 0:
        # Seed a new "minimum component"
        num += 1
        # Grab a random unused node n1
        #idx1 = np.random.randint(0, len(unused))
        idx1 = 0
        n1 = unused[idx1]
        unused.remove(n1)
        neigbs = list(g_pos.neighbors(n1))
        neigbs = ut.isect(neigbs, unused)
        while len(neigbs) > 0:
            # Find node n2, that n1 could be connected to
            #idx2 = np.random.randint(0, len(neigbs))
            idx2 = 0
            n2 = neigbs[idx2]
            unused.remove(n2)
            # Collapse negative information of n1 and n2
            g_neg = nx.contracted_nodes(g_neg, n1, n2)
            # Compute new possible positive edges
            g_pos = nx.complement(g_neg)
            # Iterate until n1 has no more possible connections
            neigbs = list(g_pos.neighbors(n1))
            neigbs = ut.isect(neigbs, unused)
    print('num = %r' % (num,))
    return num


def nx_mincut_edges_weighted(G, s, t, capacity='weight'):
    # http://stackoverflow.com/questions/33332462/minimum-s-t-edge-cut-which-takes-edge-weight-into-consideration
    cut_weight, partitions = nx.minimum_cut(G, s, t, capacity=capacity)
    edge_cut_list = []
    for p1_node in partitions[0]:
        for p2_node in partitions[1]:
            if G.has_edge(p1_node, p2_node):
                edge_cut_list.append((p1_node, p2_node))
    # assert edge_cut_list == nx_edges_between(G, partitions[0], partitions[1])
    return edge_cut_list


def weighted_diamter(graph, weight=None):
    if weight is None:
        distances = nx.all_pairs_shortest_path_length(graph)
    else:
        distances = nx.all_pairs_dijkstra_path_length(graph, weight=weight)
    if isinstance(distances, dict):
        eccentricities = (max(list(dists.values())) for node, dists in distances.items())
    else:
        eccentricities = (max(list(dists.values())) for node, dists in distances)
    diameter = max(list(eccentricities))
    return diameter


def mincost_diameter_augment(graph, max_cost, candidates=None, weight=None, cost=None):
    r"""
    PROBLEM: Bounded Cost Minimum Diameter Edge Addition (BCMD)

    Args:
        graph (nx.Graph): input graph
        max_cost (float): maximum weighted diamter of the graph
        weight (str): key of the edge weight attribute
        cost (str): key of the edge cost attribute
        candidates (list): set of non-edges, optional, defaults
            to the complement of the graph

    Returns:
        None: if no solution exists
        list: minimum cost edges if solution exists

    Notes:
        We are given a graph G = (V, E) with an edge weight function w, an edge
        cost function c, an a maximum cost B.

        The goal is to find a set of candidate non-edges F.

        Let x[e] in {0, 1} denote if a non-edge e is excluded or included.

        minimize sum(c(e) * x[e] for e in F)
        such that
        weighted_diamter(graph.union({e for e in F if x[e]})) <= B

    References:
        https://www.cse.unsw.edu.au/~sergeg/papers/FratiGGM13isaac.pdf
        http://www.cis.upenn.edu/~sanjeev/papers/diameter.pdf
        http://dl.acm.org/citation.cfm?id=2953882

    Notes:
        There is a 4-Approximation of the BCMD problem
        Running time is O((3 ** B * B ** 3 + n + log(B * n)) * B * n ** 2)

        This algorithm usexs a clustering approach to find a set C, of B + 1
        cluster centers.  Then we create a minimum height rooted tree, T = (U
        \subseteq V, D) so that C \subseteq U.  This tree T approximates an
        optimal B-augmentation.

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_graph import *  # NOQA
        >>> import utool as ut
        >>> graph = nx.Graph()
        >>> if nx.__version__.startswith('1'):
        >>>     nx.add_path = nx.Graph.add_path
        >>> nx.add_path(graph, range(6))
        >>> #cost_func   = lambda e: e[0] + e[1]
        >>> cost_func   = lambda e: 1
        >>> weight_func = lambda e: (e[0]) / e[1]
        >>> comp_graph = nx.complement(graph)
        >>> nx.set_edge_attributes(graph, name='cost', values={e: cost_func(e) for e in graph.edges()})
        >>> nx.set_edge_attributes(graph, name='weight', values={e: weight_func(e) for e in graph.edges()})
        >>> nx.set_edge_attributes(comp_graph, name='cost', values={e: cost_func(e) for e in comp_graph.edges()})
        >>> nx.set_edge_attributes(comp_graph, name='weight', values={e: weight_func(e) for e in comp_graph.edges()})
        >>> candidates = list(comp_graph.edges(data=True))
        >>> max_cost = 2
        >>> cost = 'cost'
        >>> weight = 'weight'
        >>> best_edges = mincost_diameter_augment(graph, max_cost, candidates, weight, cost)
        >>> print('best_edges = %r' % (best_edges,))
        >>> soln_edges = greedy_mincost_diameter_augment(graph, max_cost, candidates, weight, cost)
        >>> print('soln_edges = %r' % (soln_edges,))
    """
    import utool as ut
    import operator as op

    if candidates is None:
        candidates = list(graph.complement().edges(data=True))

    def augment_add(graph, edges):
        aug_graph = graph.copy()
        aug_graph.add_edges_from(edges)
        return aug_graph

    def solution_energy(chosen_edges):
        if weight is None:
            return len(chosen_edges)
        else:
            return sum(d[weight] for (u, v, d) in chosen_edges)

    variable_basis = [(0, 1) for _ in candidates]
    best_energy = np.inf
    best_soln = None

    soln_generator = ut.product(*variable_basis)
    length = reduce(op.mul, map(len, variable_basis), 1)
    if length > 3000:
        # Let the user know that it might take some time to find a solution
        soln_generator = ut.ProgIter(soln_generator, label='BruteForce BCMD',
                                     length=length)
    # Brute force solution
    for x in soln_generator:
        chosen_edges = ut.compress(candidates, x)
        aug_graph = augment_add(graph, chosen_edges)
        total_cost = weighted_diamter(aug_graph, weight=cost)
        energy = solution_energy(chosen_edges)
        if total_cost <= max_cost:
            if energy < best_energy:
                best_energy = energy
                best_soln = x

    best_edges = ut.compress(candidates, best_soln)
    return best_edges


def greedy_mincost_diameter_augment(graph, max_cost, candidates=None, weight=None, cost=None):
    # import utool as ut

    def solution_cost(graph):
        return weighted_diamter(graph, weight=cost)

    def solution_energy(chosen_edges):
        if weight is None:
            return len(chosen_edges)
        else:
            return sum(d[weight] for (u, v, d) in chosen_edges)

    def augment_add(graph, edges):
        aug_graph = graph.copy()
        aug_graph.add_edges_from(edges)
        return aug_graph

    def augment_remove(graph, edges):
        aug_graph = graph.copy()
        aug_graph.remove_edges_from(edges)
        return aug_graph

    base_cost = solution_cost(graph)
    # base_energy = 0

    full_graph = augment_add(graph, candidates)
    full_cost = solution_cost(full_graph)
    # full_energy = solution_energy(candidates)

    def greedy_improvement(soln_graph, available_candidates, base_cost=None):
        """
        Choose edge that results in the best improvement
        """
        best_loss = None
        best_cost = None
        best_energy = None
        best_e = None
        best_graph = None

        for e in available_candidates:
            aug_graph = augment_add(soln_graph, [e])
            aug_cost = solution_cost(aug_graph)
            aug_energy = solution_energy([e])

            # We don't want to go over if possible
            aug_loss = max(aug_cost - max_cost, 0)

            if best_loss is None or aug_loss <= best_loss:
                if best_energy is None or aug_energy < best_energy:
                    best_loss = aug_loss
                    best_e = e
                    best_graph = aug_graph
                    best_cost = aug_cost
                    best_energy = aug_energy

        if best_e is None:
            return None
        else:
            return best_cost, best_graph, best_energy, best_e

    import warnings
    if full_cost > max_cost:
        warnings.warn('no feasible solution')
    else:
        soln_graph = graph.copy()
        available_candidates = candidates[:]
        soln_edges = []
        soln_energy = 0
        soln_cost = base_cost

        # Add edges to the solution until the cost is feasible
        while soln_cost > max_cost and len(available_candidates):
            tup = greedy_improvement(soln_graph, available_candidates, soln_cost)
            if tup is None:
                warnings.warn('no improvement found')
                break
            soln_cost, soln_graph, best_energy, best_e = tup
            soln_energy += best_energy
            soln_edges.append(best_e)
            available_candidates.remove(best_e)

        # Check to see we can remove edges while maintaining feasibility
        for e in soln_edges[:]:
            aug_graph = augment_remove(soln_graph, [e])
            aug_cost = solution_cost(aug_graph)
            if aug_cost <= soln_cost:
                soln_cost = aug_cost
                soln_graph = aug_graph
                soln_edges.remove(e)

    return soln_edges


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_graph --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
