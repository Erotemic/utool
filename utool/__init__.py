"""
UTool - Useful Utility Tools
   Your friendly neighborhood utility tools

TODO: INSERT APACHE VERSION 2.0 LICENCE IN ALL FILES
(Although it should be implied that the entire module and repo is released under
that licence.)

pip install git+https://github.com/Erotemic/utool.git@next
"""
# Utool is released under the Apache License Version 2.0

import sys

__version__ = '2.1.7'

__DYNAMIC__ = True
if __DYNAMIC__:
    __DYNAMIC__ = '--nodyn' not in sys.argv
else:
    __DYNAMIC__ = '--dyn' in sys.argv
# THESE COMMANDS WILL WRITE THE IMPORT FILE
"""
python -c "import utool" --dump-utool-init
python -c "import utool" --print-utool-init --dyn
python -c "import utool" --update-utool-init --dyn
"""

__VERYVERBOSE__ = '--veryverbose' in sys.argv or '--very-verbose' in sys.argv
__PRINT_INJECT_ORDER__ = __VERYVERBOSE__ or '--print-inject-order' in sys.argv

# HAVE TO HACK THIS IN FOR UTOOL.__INIT__ ONLY
# OTHER MODULE CAN USE NOINJECT
if __PRINT_INJECT_ORDER__:
    # TODO METAUTIL
    from six.moves import builtins
    from utool._internal import meta_util_dbg
    callername = meta_util_dbg.get_caller_name(N=1, strict=False)
    fmtdict = dict(callername=callername, modname='utool.__init__')
    msg = '[util_inject] {modname} is imported by {callername}'.format(**fmtdict)
    builtins.print(msg)


IMPORT_TUPLES = [
    ('_internal',      None),
    ('util_alg',       ['cartesian', 'almost_eq']),
    ('util_aliases',   ['ddict', 'odict']),
    ('util_arg',       ['get_argval', 'get_argflag', 'argv_flag_dec', 'QUIET',
                        'VERBOSE']),
    ('util_assert',    None),
    ('util_autogen',   None),
    ('util_cache',     ['global_cache_read', 'global_cache_write']),
    ('util_cplat',     ['cmd', 'view_directory']),
    ('util_class',     None),
    ('util_const',     None),
    ('util_csv',       None),
    ('util_config',    None),
    ('util_dbg',       ['execstr_dict', 'save_testdata', 'load_testdata',
                        'get_caller_name', 'import_testdata', 'embed',
                        'quitflag', 'inIPython', 'printvar2', 'all_rrr']),
    ('util_dev',       ['get_stats_str', 'get_stats', 'myprint',
                        'get_object_nbytes']),
    ('util_decor',     ['ignores_exc_tb', 'indent_func', 'accepts_numpy',
                        'accepts_scalar_input_vector_output',
                        'accepts_scalar_input']),
    ('util_dict',      None),
    ('util_func',      None),
    ('util_grabdata',  None),
    ('util_gridsearch',  None),
    ('util_git',       None),
    ('util_latex',     None),
    ('util_graph',   None),
    ('util_hash',      ['hashstr_arr', 'hashstr']),
    ('util_import',    None),
    ('util_inject',    None),
    ('util_io',        None),
    ('util_iter',      ['iflatten', 'ichunks']),
    ('util_inspect',   None),
    ('util_ipynb',     None),
    ('util_logging',   None),
    ('util_list',      ['alloc_lists', 'list_index', 'npfind', 'index_of',
                        'flatten']),
    ('util_num',       None),
    ('util_numpy',       None),
    ('util_path',      ['checkpath', 'ensuredir', 'assertpath', 'truepath',
                        'list_images', 'copy', 'glob', 'grep']),
    ('util_print',     ['horiz_print', 'printshape', 'Indenter']),
    ('util_progress',  ['progress_func']),
    ('util_profile',   None),
    ('util_project',   None),
    ('util_parallel',  None),
    ('util_resources', ['print_resource_usage']),
    ('util_str',       ['byte_str2', 'horiz_string', 'theta_str']),
    ('util_sysreq',    None),
    ('util_sqlite',    None),
    ('util_setup',     'presetup'),
    ('util_set',       None),
    ('util_regex',     ['regex_search']),
    ('util_time',      ['tic', 'toc', 'Timer']),
    ('util_tags',      None),
    ('util_type',      None),
    ('util_tests',     None),
    ('util_web',     None),
    ('DynamicStruct',  ['DynStruct']),
    ('Preferences',    ['Pref']),
    ]


DOELSE = False

if __DYNAMIC__:
    # TODO: import all utool external prereqs. Then the imports will not import
    # anything that has already in a toplevel namespace
    # COMMENTED OUT FOR FROZEN __INIT__
    # Dynamically import listed util libraries and their members.
    from utool._internal import util_importer
    # FIXME: this might actually work with rrrr, but things arent being
    # reimported because they are already in the modules list
    import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES)
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    # <AUTOGEN_INIT>

    from utool import _internal
    from utool import util_alg
    from utool import util_aliases
    from utool import util_arg
    from utool import util_assert
    from utool import util_autogen
    from utool import util_cache
    from utool import util_cplat
    from utool import util_class
    from utool import util_const
    from utool import util_csv
    from utool import util_config
    from utool import util_dbg
    from utool import util_dev
    from utool import util_decor
    from utool import util_dict
    from utool import util_func
    from utool import util_grabdata
    from utool import util_gridsearch
    from utool import util_git
    from utool import util_latex
    from utool import util_graph
    from utool import util_hash
    from utool import util_import
    from utool import util_inject
    from utool import util_io
    from utool import util_iter
    from utool import util_inspect
    from utool import util_ipynb
    from utool import util_logging
    from utool import util_list
    from utool import util_num
    from utool import util_numpy
    from utool import util_path
    from utool import util_print
    from utool import util_progress
    from utool import util_profile
    from utool import util_project
    from utool import util_parallel
    from utool import util_resources
    from utool import util_str
    from utool import util_sysreq
    from utool import util_sqlite
    from utool import util_setup
    from utool import util_set
    from utool import util_regex
    from utool import util_time
    from utool import util_tags
    from utool import util_type
    from utool import util_tests
    from utool import util_web
    from utool import DynamicStruct
    from utool import Preferences

    from utool.util_alg import (FOOT_PER_MILE, HAVE_NUMPY, HAVE_SCIPY,
                                KM_PER_MILE, MM_PER_INCH, PHI, PHI_A, PHI_B,
                                TAU, absdiff, almost_allsame, almost_eq,
                                apply_grouping, bayes_rule, choose,
                                colwise_diag_idxs, compare_groups, cumsum,
                                defaultdict, deg_to_rad, diagonalized_iter,
                                edit_distance, enumerate_primes,
                                euclidean_dist, expensive_task_gen, factors,
                                fibonacci, fibonacci_approx,
                                fibonacci_iterative, fibonacci_recursive,
                                find_group_consistencies,
                                find_group_differences,
                                flatten_membership_mapping, generate_primes,
                                get_nth_bell_number, get_nth_prime,
                                get_nth_prime_bruteforce, get_phi,
                                get_phi_ratio1, get_prime_index,
                                greedy_max_inden_setcover, group_indices,
                                grouping_delta, grouping_delta_stats,
                                iapply_grouping, inbounds, is_prime, item_hist,
                                knapsack, knapsack_greedy, knapsack_ilp,
                                knapsack_iterative, knapsack_iterative_int,
                                knapsack_iterative_numpy, knapsack_recursive,
                                longest_common_substring,
                                max_size_max_distance_subset,
                                maximin_distance_subset1d,
                                maximum_distance_subset,
                                negative_minclamp_inplace, norm_zero_one,
                                normalize, num_partitions, number_of_decimals,
                                prod, product_nonsame, product_nonsame_self,
                                rad_to_deg, safe_div, safe_pdist, self_prodx,
                                setcover_greedy, setcover_ilp, solve_boolexpr,
                                square_pdist, standardize_boolexpr,
                                triangular_number, ungroup, ungroup_gen,
                                ungroup_unique, unixtime_hourdiff,
                                upper_diag_self_prodx, xywh_to_tlbr,)
    from utool.util_aliases import (OrderedDict, combinations, ddict, icomb,
                                    iprod, namedtuple, odict, partial,
                                    product,)
    from utool.util_arg import (ArgumentParser2, DEBUG, DEBUG2, NOT_QUIET,
                                NO_ASSERTS, QUIET, REPORT, SAFE, SILENT,
                                STRICT, SUPER_STRICT, TRACE, USE_ASSERT,
                                VERBOSE, VERBOSE_ARGPARSE, VERYVERBOSE,
                                argflag, argparse_dict, argv_flag_dec,
                                argv_flag_dec_true, argval, aug_sysargv,
                                autogen_argparse2, autogen_argparse_block,
                                get_arg, get_arg_dict, get_argflag,
                                get_argv_tail, get_argval, get_cmdline_varargs,
                                get_dict_vals_from_commandline, get_flag,
                                get_fpath_args, get_module_verbosity_flags,
                                get_varargs, get_verbflag, make_argparse2,
                                parse_arglist_hack, parse_cfgstr_list,
                                parse_dict_from_argv, reset_argrecord,
                                switch_sanataize,)
    from utool.util_assert import (assert_all_eq, assert_all_in,
                                   assert_all_not_None, assert_almost_eq,
                                   assert_eq, assert_eq_len, assert_inbounds,
                                   assert_lessthan, assert_lists_eq,
                                   assert_raises, assert_same_len,
                                   assert_scalar_list, assert_unflat_level,
                                   assert_unique, get_first_None_position,
                                   lists_eq,)
    from utool.util_autogen import (PythonStatement, auto_docstr,
                                    autofix_codeblock, deque,
                                    dump_autogen_code,
                                    find_modname_in_pythonpath,
                                    is_modname_in_pythonpath,
                                    load_func_from_module, make_args_docstr,
                                    make_cmdline_docstr, make_default_docstr,
                                    make_default_module_maintest,
                                    make_docstr_block, make_example_docstr,
                                    make_returns_or_yeilds_docstr, makeinit,
                                    print_auto_docstr,
                                    remove_codeblock_syntax_sentinals,
                                    write_modscript_alias,)
    from utool.util_cache import (Cachable, CacheMissException, Cacher,
                                  GlobalShelfContext, KeyedDefaultDict,
                                  LRUDict, LazyDict, LazyList, ShelfCacher,
                                  USE_CACHE, VERBOSE_CACHE, cached_func,
                                  cachestr_repr, chain, consensed_cfgstr,
                                  delete_global_cache, from_json,
                                  get_cfgstr_from_args, get_default_appname,
                                  get_func_result_cachekey,
                                  get_global_cache_dir, get_global_shelf_fpath,
                                  get_lru_cache, global_cache_dump,
                                  global_cache_read, global_cache_write,
                                  load_cache, make_utool_json_encoder,
                                  save_cache, shelf_open, text_dict_read,
                                  text_dict_write, time_different_diskstores,
                                  to_json, tryload_cache, tryload_cache_list,
                                  tryload_cache_list_with_compute,
                                  view_global_cache_dir,)
    from utool.util_cplat import (COMPUTER_NAME, DARWIN, EXIT_FAILURE,
                                  EXIT_SUCCESS, HAVE_PATHLIB, LIB_DICT,
                                  LIB_EXT_LIST, LINUX, OS_TYPE, PYLIB_DICT,
                                  UNIX, WIN32, assert_installed_debian,
                                  change_term_title, check_installed_debian,
                                  chmod, chmod_add_executable, cmd, cmd2,
                                  editfile, ensure_app_cache_dir,
                                  ensure_app_resource_dir, get_app_cache_dir,
                                  get_app_resource_dir, get_computer_name,
                                  get_dir_diskspaces, get_disk_space,
                                  get_dynamic_lib_globstrs,
                                  get_dynlib_dependencies, get_dynlib_exports,
                                  get_file_info, get_file_nBytes,
                                  get_file_nBytes_str, get_flops,
                                  get_free_diskbytes, get_install_dirs,
                                  get_lib_ext, get_path_dirs,
                                  get_plat_specifier, get_pylib_ext,
                                  get_python_dynlib, get_resource_dir,
                                  get_system_python_library,
                                  get_total_diskbytes, get_user_name,
                                  geteditor, getroot, in_pyinstaller_package,
                                  ipython_paste, is64bit_python,
                                  is_file_executable, is_file_writable,
                                  ls_libs, pip_install, platform_cache_dir,
                                  print_dir_diskspace, print_path,
                                  print_system_users, python_executable,
                                  quote_single_command, run_realtime_process,
                                  search_env_paths, send_keyboard_input,
                                  set_process_title, shell,
                                  spawn_delayed_ipython_paste, startfile,
                                  truepath, unixpath, unload_module, vd,
                                  view_directory, view_file_in_directory,)
    from utool.util_class import (HashComparable, HashComparable2,
                                  HashComparableMetaclass, KwargsWrapper,
                                  QUIET_CLASS, ReloadingMetaclass,
                                  VERBOSE_CLASS,
                                  autogen_explicit_injectable_metaclass,
                                  autogen_import_list, compare_instance,
                                  decorate_class_method, decorate_postinject,
                                  get_classname, get_comparison_methods,
                                  get_comparison_operators, get_funcglobals,
                                  get_funcname, get_injected_modules,
                                  get_method_func, inject_all_external_modules,
                                  inject_func_as_method,
                                  inject_func_as_property,
                                  inject_func_as_unbound_method,
                                  inject_instance, makeForwardingMetaclass,
                                  make_class_method_decorator,
                                  make_class_postinject_decorator,
                                  postinject_instance, reload_class,
                                  reload_class_methods,
                                  reload_injected_modules, reloadable_class,
                                  reloading_meta_metaclass_factory,
                                  remove_private_obfuscation,
                                  test_reloading_metaclass,)
    from utool.util_const import (NoParam,)
    from utool.util_csv import (CSV, make_csv_table, make_standard_csv,
                                numpy_to_csv, read_csv,)
    from utool.util_config import (get_default_global_config,
                                   get_default_repo_config, read_repo_config,
                                   write_default_repo_config,)
    from utool.util_dbg import (COLORED_EXCEPTIONS, EmbedOnException, FORCE_TB,
                                RAISE_ALL, TB, breakpoint, debug_exception,
                                debug_hstack, debug_list, debug_npstack,
                                debug_vstack, embed, embed2,
                                embed_on_exception_context, eoxc,
                                execstr_attr_list, execstr_dict, execstr_func,
                                execstr_parent_locals, explore_module,
                                explore_stack, fix_embed_globals, fmtlocals,
                                formatex, get_caller_lineno,
                                get_caller_modname, get_caller_name,
                                get_caller_prefix, get_current_stack_depth,
                                get_localvar_from_stack, get_parent_frame,
                                get_reprs, get_stack_frame, get_var_from_stack,
                                get_varname_from_locals,
                                get_varname_from_stack, get_varstr,
                                get_varval_from_locals, haveIPython,
                                import_testdata, inIPython,
                                in_jupyter_notebook, ipython_execstr,
                                is_valid_varname, load_testdata,
                                module_functions, parse_locals_keylist,
                                print_frame, print_keys, print_traceback,
                                printex, printvar, printvar2,
                                public_attributes, qflag, quasiquote, quit,
                                quitflag, save_testdata,
                                search_stack_for_localvar,
                                search_stack_for_var, split, super_print,
                                varname_regex,)
    from utool.util_dev import (ClassAttrDictProxy, ColumnLists,
                                DEVELOPER_MODE, DictLike_old, ENABLE_MEMTRACK,
                                INDEXABLE_TYPES, InteractiveIter,
                                InteractivePrompt, LIVE_INTERACTIVE_ITER,
                                MemoryTracker, NamedPartial, NiceRepr,
                                PriorityQueue, STAT_KEY_ORDER, Shortlist,
                                USER_MODE, are_you_sure, autopep8_diff,
                                compile_cython, copy_text_to_clipboard,
                                delayed_retry_gen, dev_ipython_copypaster,
                                disable_garbage_collection,
                                enable_garbage_collection, ensure_str_list,
                                ensureqt, exec_argparse_funckw, exec_funckw,
                                execstr_funckw, extract_timeit_setup, find_exe,
                                fix_super_reload, focusvim, garbage_collect,
                                get_clipboard, get_cython_exe,
                                get_dev_paste_code, get_jagged_stats,
                                get_nonconflicting_path_old,
                                get_nonconflicting_string, get_object_base,
                                get_object_nbytes, get_object_size_str,
                                get_overlaps, get_partial_func_name,
                                get_setdiff_info, get_setdiff_items,
                                get_statdict, get_stats, get_stats_str,
                                get_submodules_from_dpath, grace_period,
                                init_catch_ctrl_c, input_timeout, instancelist,
                                inverable_group_multi_list,
                                inverable_unique_two_lists, ipcopydev,
                                is_developer, iup, make_at_least_n_items_valid,
                                make_call_graph, make_object_graph,
                                memory_dump, numpy_list_num_bits,
                                overrideable_partial, pandas_reorder,
                                print_object_size, print_object_size_tree,
                                pylab_qt4, report_memsize, reset_catch_ctrl_c,
                                search_module, search_utool, set_clipboard,
                                set_overlap_items, set_overlaps, stats_dict,
                                strip_line_comments, super2, timeit_compare,
                                timeit_grid, tuples_to_unique_scalars,
                                uninvert_unique_two_lists,
                                user_cmdline_prompt,)
    from utool.util_decor import (IGNORE_TRACEBACK, NOINDENT_DECOR,
                                  ONEX_REPORT_INPUT, PROFILING, SIG_PRESERVE,
                                  UNIQUE_NUMPY, accepts_numpy,
                                  accepts_scalar_input, accepts_scalar_input2,
                                  accepts_scalar_input_vector_output,
                                  apply_docstr, classproperty,
                                  debug_function_exceptions, dummy_args_decor,
                                  getter_1to1, getter_1toM, ignores_exc_tb,
                                  indent_func, interested, lazyfunc, memoize,
                                  memoize_nonzero, memoize_single,
                                  memoize_zero, on_exception_report_input,
                                  preserve_sig, show_return_value,
                                  test_ignore_exec_traceback, time_func,
                                  tracefunc, tracefunc_xml,)
    from utool.util_dict import (AutoDict, AutoOrderedDict, AutoVivification,
                                 DefaultValueDict, DictLike,
                                 OrderedAutoVivification,
                                 all_dict_combinations,
                                 all_dict_combinations_lbls,
                                 all_dict_combinations_ordered,
                                 assert_keys_are_subset, augdict,
                                 build_conflict_dict, count_dict_vals,
                                 delete_dict_keys, delete_keys, depth_atleast,
                                 dict_accum, dict_assign, dict_filter_nones,
                                 dict_find_keys, dict_find_other_sameval_keys,
                                 dict_hist, dict_hist_cumsum,
                                 dict_intersection, dict_isect,
                                 dict_isect_combine, dict_keysubset,
                                 dict_set_column, dict_setdiff, dict_stack,
                                 dict_stack2, dict_subset, dict_take,
                                 dict_take_asnametup, dict_take_column,
                                 dict_take_gen, dict_take_list, dict_take_pop,
                                 dict_to_keyvals, dict_union, dict_union2,
                                 dict_union3, dict_union_combine,
                                 dict_update_newkeys, dict_where_len0,
                                 dictinfo, dzip, flatten_dict_items,
                                 flatten_dict_vals, get_dict_column,
                                 get_dict_hashid, group_items, group_pairs,
                                 groupby_attr, groupby_tags, hashdict,
                                 hierarchical_group_items,
                                 hierarchical_map_vals, hmap_vals,
                                 iflatten_dict_values, invert_dict, is_dicteq,
                                 iter_all_dict_combinations_ordered,
                                 iteritems_sorted, keys_sorted_by_value,
                                 map_dict_keys, map_dict_vals, map_keys,
                                 map_vals, merge_dicts, move_odict_item,
                                 order_dict_by, range_hist, sort_dict,
                                 update_dict, update_existing,)
    from utool.util_func import (compose_functions, identity,)
    from utool.util_grabdata import (BadZipfile, HASH_DICT, TESTIMG_URL_DICT,
                                     archive_files, clean_dropbox_link,
                                     clear_test_img_cache, download_url,
                                     experiment_download_multiple_urls,
                                     geo_locate, get_file_local_hash,
                                     get_prefered_browser,
                                     get_valid_test_imgkeys,
                                     grab_file_remote_hash, grab_file_url,
                                     grab_s3_contents,
                                     grab_selenium_chromedriver,
                                     grab_selenium_driver, grab_test_imgpath,
                                     grab_zipped_url, list_remote,
                                     open_url_in_browser, read_s3_contents,
                                     rsync, s3_dict_encode_to_str,
                                     s3_str_decode_to_dict, scp_pull,
                                     split_archive_ext, unarchive_file,
                                     untar_file, unzip_file, url_read,
                                     url_read_text,)
    from utool.util_gridsearch import (CountstrParser, DimensionBasis,
                                       GridSearch, INTERNAL_CFGKEYS,
                                       NAMEVARSEP, Nesting, ParamInfo,
                                       ParamInfoBool, ParamInfoList,
                                       constrain_cfgdict_list,
                                       customize_base_cfg, get_cfg_lbl,
                                       get_cfgdict_lbl_list_subset,
                                       get_cfgdict_list_subset,
                                       get_nonvaried_cfg_lbls,
                                       get_varied_cfg_lbls,
                                       grid_search_generator, gridsearch_timer,
                                       interact_gridsearch_result_images,
                                       lookup_base_cfg_list, make_cfglbls,
                                       make_constrained_cfg_and_lbl_list,
                                       noexpand_parse_cfgstrs, parse_argv_cfg,
                                       parse_cfgstr3, parse_cfgstr_list2,
                                       parse_cfgstr_name_options,
                                       parse_nestings, parse_nestings2,
                                       partition_varied_cfg_list,
                                       recombine_nestings,
                                       testdata_grid_search,)
    from utool.util_git import (Repo, RepoManager, git_sequence_editor_squash,
                                isdir, std_build_command,)
    from utool.util_latex import (compile_latex_text, compress_pdf,
                                  convert_pdf_to_image, ensure_colvec,
                                  ensure_rowvec, escape_latex,
                                  find_ghostscript_exe, get_latex_figure_str,
                                  get_latex_figure_str2, is_substr,
                                  latex_get_stats, latex_multicolumn,
                                  latex_multirow, latex_newcommand,
                                  latex_sanitize_command_name, latex_scalar,
                                  long_substr, make_full_document,
                                  make_score_tabular, make_stats_tabular,
                                  render_latex, render_latex_text, replace_all,
                                  tabular_join,)
    from utool.util_graph import (all_multi_paths, approx_min_num_components,
                                  bfs_conditional, bfs_multi_edges,
                                  color_nodes, dag_longest_path,
                                  dfs_conditional, dict_depth,
                                  edges_to_adjacency_list, get_allkeys,
                                  get_graph_bounding_box, get_levels,
                                  graph_info, greedy_mincost_diameter_augment,
                                  longest_levels, mincost_diameter_augment,
                                  nx_all_nodes_between,
                                  nx_all_simple_edge_paths,
                                  nx_common_ancestors, nx_common_descendants,
                                  nx_contracted_nodes, nx_dag_node_rank,
                                  nx_delete_None_edge_attr,
                                  nx_delete_None_node_attr,
                                  nx_delete_edge_attr, nx_delete_node_attr,
                                  nx_edges, nx_edges_between,
                                  nx_ensure_agraph_color, nx_from_adj_dict,
                                  nx_from_matrix, nx_from_node_edge,
                                  nx_gen_edge_attrs, nx_gen_edge_values,
                                  nx_gen_node_attrs, nx_gen_node_values,
                                  nx_get_default_edge_attributes,
                                  nx_get_default_node_attributes,
                                  nx_make_adj_matrix, nx_mincut_edges_weighted,
                                  nx_minimum_weight_component, nx_node_dict,
                                  nx_set_default_edge_attributes,
                                  nx_set_default_node_attributes,
                                  nx_sink_nodes, nx_source_nodes,
                                  nx_to_adj_dict, nx_topsort_nodes,
                                  nx_topsort_rank, nx_transitive_reduction,
                                  paths_to_root, reverse_path,
                                  reverse_path_edges, shortest_levels,
                                  simplify_graph, stack_graphs,
                                  subgraph_from_edges, testdata_graph,
                                  translate_graph, translate_graph_to_origin,
                                  traverse_path, weighted_diamter,)
    from utool.util_hash import (ALPHABET, ALPHABET_16, ALPHABET_27,
                                 ALPHABET_41, BIGBASE, DictProxyType, HASH_LEN,
                                 HASH_LEN2, SEP_BYTE, SEP_STR, augment_uuid, b,
                                 combine_hashes, combine_uuids,
                                 convert_bytes_to_bigbase,
                                 convert_hexstr_to_bigbase, digest_data,
                                 freeze_hash_bytes, get_file_hash,
                                 get_file_uuid, get_zero_uuid, hash_data,
                                 hashable_to_uuid, hashid_arr, hashstr,
                                 hashstr27, hashstr_arr, hashstr_arr27,
                                 hashstr_md5, hashstr_sha1, image_uuid,
                                 make_hash, random_nonce, random_uuid,
                                 stringlike, write_hash_file,
                                 write_hash_file_for_path,)
    from utool.util_import import (check_module_installed,
                                   get_modpath_from_modname, import_modname,
                                   import_module_from_fpath, import_star,
                                   import_star_execstr, package_contents,
                                   possible_import_patterns, tryimport,)
    from utool.util_inject import (ARGV_DEBUG_FLAGS, DEBUG_PRINT,
                                   DEBUG_PRINT_N, DEBUG_SLOW_IMPORT,
                                   DUMMYPROF_FUNC, EXIT_ON_INJECT_MODNAME,
                                   PRINT_INJECT_ORDER, PROFILE_FUNC,
                                   PROF_FUNC_PAT_LIST, PROF_MOD_PAT_LIST,
                                   TIMERPROF_FUNC, argv,
                                   colored_pygments_excepthook, inject,
                                   inject2, inject_colored_exceptions,
                                   inject_print_functions, inject_python_code,
                                   inject_python_code2, make_module_print_func,
                                   make_module_profile_func,
                                   make_module_reload_func,
                                   make_module_write_func, memprof, noinject,
                                   reload_module,
                                   split_python_text_into_lines,)
    from utool.util_io import (HAS_H5PY, HAS_NUMPY, HAVE_LOCKFILE, load_cPkl,
                               load_data, load_hdf5, load_json, load_numpy,
                               load_pytables, load_text, lock_and_load_cPkl,
                               lock_and_save_cPkl, read_from, read_lines_from,
                               readfrom, save_cPkl, save_data, save_hdf5,
                               save_json, save_numpy, save_pytables, save_text,
                               try_decode, write_to, writeto,)
    from utool.util_iter import (and_iters, ensure_iterable,
                                 evaluate_generator, ichunk_slices, ichunks,
                                 ichunks_cycle, ichunks_list, ichunks_noborder,
                                 ichunks_replicate, ifilter_Nones,
                                 ifilter_items, ifilterfalse_items, iflatten,
                                 iget_list_column, iget_list_column_slice,
                                 interleave, isiterable, isscalar,
                                 itake_column, iter_compress, iter_multichunks,
                                 iter_window, itertwo, next_counter,
                                 random_combinations, random_product,
                                 wrap_iterable,)
    from utool.util_inspect import (BaronWraper, KWReg, LIB_PATH,
                                    VERBOSE_INSPECT, VERYVERB_INSPECT,
                                    argparse_funckw, check_module_usage,
                                    check_static_member_vars, dummy_func,
                                    exec_func_doctest, exec_func_src,
                                    exec_func_src2, execstr_func_doctest,
                                    filter_valid_kwargs,
                                    find_funcs_called_with_kwargs,
                                    get_argnames, get_dev_hints, get_docstr,
                                    get_func_argspec, get_func_docblocks,
                                    get_func_kwargs, get_func_modname,
                                    get_func_sourcecode, get_funcdoc,
                                    get_funcfpath, get_funckw,
                                    get_funcnames_from_modpath,
                                    get_instance_attrnames,
                                    get_internal_call_graph, get_kwargs,
                                    get_kwdefaults, get_kwdefaults2,
                                    get_module_from_class,
                                    get_module_owned_functions,
                                    get_object_methods, get_unbound_args,
                                    help_members,
                                    infer_arg_types_and_descriptions,
                                    infer_function_info, inherit_kwargs,
                                    is_bateries_included, is_defined_by_module,
                                    is_defined_by_module2, is_valid_python,
                                    iter_module_doctestable,
                                    list_class_funcnames,
                                    list_global_funcnames,
                                    lookup_attribute_chain,
                                    parse_func_kwarg_keys,
                                    parse_function_names, parse_import_names,
                                    parse_kwarg_keys, parse_return_type,
                                    prettyprint_parsetree,
                                    recursive_parse_kwargs, set_funcdoc,
                                    set_funcname,
                                    special_parse_process_python_code,
                                    zzz_profiled_is_no, zzz_profiled_is_yes,)
    from utool.util_ipynb import (IPYNBCell, code_cell, export_notebook,
                                  format_cells, make_autogen_str,
                                  make_notebook, markdown_cell,
                                  normalize_cells, repr_single_for_md,
                                  run_ipython_notebook,)
    from utool.util_logging import (CustomStreamHandler, LOGGING_VERBOSE,
                                    PRINT_ALL_CALLERS, add_logging_handler,
                                    debug_logging_iostreams, ensure_logging,
                                    get_current_log_fpath,
                                    get_current_log_text, get_log_fpath,
                                    get_logging_dir, get_shelves_dir,
                                    get_utool_logger, is_logging,
                                    logdir_cacheid, start_logging,
                                    stop_logging, testlogprog,)
    from utool.util_list import (accumulate, alloc_lists, alloc_nones, allsame,
                                 and_lists, argmax, argmin, argsort, argsort2,
                                 aslist, broadcast_zip, bzip, ceil, compress,
                                 debug_consec_list, debug_duplicate_items,
                                 delete_items_by_index, delete_list_items,
                                 depth, depth_profile, duplicates_exist, emap,
                                 ensure_list_size, equal, estarmap, ezip,
                                 filter_Nones, filter_items, filter_startswith,
                                 filterfalse_items, find_duplicate_items,
                                 find_list_indexes, find_nonconsec_values,
                                 flag_None_items, flag_not_None_items,
                                 flag_percentile_parts, flag_unique_items,
                                 flat_unique, flatten, floor, get_dirty_items,
                                 get_list_column, get_list_column_slice,
                                 group_consecutives, group_consecutives_numpy,
                                 iflag_unique_items, index_complement,
                                 index_to_boolmask, insert_values,
                                 intersect_ordered, invertible_flatten1,
                                 invertible_flatten2,
                                 invertible_flatten2_numpy,
                                 invertible_total_flatten, is_subset,
                                 is_subset_of_any, is_superset, isdisjoint,
                                 isect, isect_indices, isetdiff_flags,
                                 issorted, issubset, issuperset, isunique,
                                 length_hint, listT, list_alignment,
                                 list_all_eq_to, list_argmax, list_argmaxima,
                                 list_argsort, list_compress,
                                 list_compresstake, list_cover,
                                 list_deep_types, list_depth, list_getattr,
                                 list_intersection, list_inverse_take,
                                 list_isdisjoint, list_issubset,
                                 list_issuperset, list_replace, list_reshape,
                                 list_roll, list_set_equal, list_strip,
                                 list_take, list_transpose, list_type,
                                 list_type_profile, list_union, list_where,
                                 list_zipcompress, list_zipflatten,
                                 list_ziptake, listclip, listfind, lmap,
                                 lstarmap, lzip, make_index_lookup,
                                 make_sortby_func, maplen, multi_replace,
                                 none_take, not_list, or_lists,
                                 partial_imap_1to1, partial_order,
                                 print_duplicate_map, priority_argsort,
                                 priority_sort, rebase_labels,
                                 recursive_replace, replace_nones,
                                 safe_listget, safe_slice, safeapply, safelen,
                                 sample_lists, sample_zip, scalar_input_map,
                                 search_list, setdiff, setdiff_flags,
                                 setdiff_ordered, setintersect,
                                 setintersect_ordered, snapped_slice, sortedby,
                                 sortedby2, strided_sample, take,
                                 take_around_percentile, take_column,
                                 take_complement, take_percentile,
                                 take_percentile_parts, total_flatten,
                                 total_unflatten, type_profile, type_profile2,
                                 type_sequence_factory, unflat_map,
                                 unflat_take, unflat_unique_rowid_map,
                                 unflat_vecmap, unflatten1, unflatten2, union,
                                 union_ordered, unique, unique_flags,
                                 unique_indices, unique_inverse,
                                 unique_ordered, unique_unordered,
                                 unpack_iterables, where, where_not_None,
                                 xor_lists, zipcompress, zipflat, ziptake,)
    from utool.util_num import (float_to_decimal, get_sys_maxfloat,
                                get_sys_maxint, get_sys_minint, int_comma_str,
                                num2_sigfig, num_fmt, order_of_magnitude_ceil,
                                sigfig_str,)
    from utool.util_numpy import (deterministic_sample, deterministic_shuffle,
                                  ensure_rng, index_of, intersect2d,
                                  make_incrementer, npfind, quantum_random,
                                  random_indexes, random_sample, sample_domain,
                                  shuffle, spaced_indexes, spaced_items,
                                  tiled_range,)
    from utool.util_path import (ChdirContext, IMG_EXTENSIONS, PRINT_CALLER,
                                 ancestor_paths, assert_exists, assertpath,
                                 augpath, basename_noext, checkpath, copy,
                                 copy_all, copy_files_to, copy_list,
                                 copy_single, delete, dirsplit,
                                 ensure_crossplat_path, ensure_ext,
                                 ensure_mingw_drive, ensure_native_path,
                                 ensure_unixslash, ensuredir, ensurefile,
                                 ensurepath, existing_commonprefix,
                                 existing_subpath, expand_win32_shortname,
                                 extend_regex, file_bytes, file_megabytes,
                                 find_lib_fpath, fnames_to_fpaths,
                                 fpath_has_ext, fpath_has_imgext,
                                 fpaths_to_fnames, get_modname_from_modpath,
                                 get_modpath, get_module_dir,
                                 get_module_subdir_list,
                                 get_nonconflicting_path, get_path_type,
                                 get_relative_modpath,
                                 get_standard_exclude_dnames,
                                 get_standard_include_patterns,
                                 get_win32_short_path_name, glob,
                                 glob_python_modules, grep, grepfile,
                                 greplines, iglob, is_module_dir,
                                 is_private_module, is_python_module, isfile,
                                 islink, ismount, list_images,
                                 longest_existing_path, ls, ls_dirs, ls_images,
                                 ls_moduledirs, ls_modulefiles,
                                 make_grep_resultstr, matching_fpaths, move,
                                 move_list, newcd, non_existing_path,
                                 num_images_in_dir, path_ndir_split,
                                 pathsplit_full, platform_path, relpath_unix,
                                 remove_broken_links, remove_dirs,
                                 remove_existing_fpaths, remove_file,
                                 remove_file_list, remove_files_in_dir,
                                 remove_fpaths, sanitize_filename,
                                 search_candidate_paths, search_in_dirs, sed,
                                 sedfile, splitdrive, symlink, tail, testgrep,
                                 touch, truepath, truepath_relative,
                                 unexpanduser, unixjoin, win_shortcut,)
    from utool.util_print import (Indenter, NO_INDENT, colorprint, cprint,
                                  dictprint, horiz_print, printNOTQUIET,
                                  printVERBOSE, printWARN, print_code,
                                  print_dict, print_difftext, print_filesize,
                                  print_list, print_locals, print_python_code,
                                  printdict, printif, printshape,)
    from utool.util_progress import (AGGROFLUSH, DEBUG_FREQ_ADJUST,
                                     FORCE_ALL_PROGRESS, NO_PROGRESS,
                                     PROGGRESS_BACKSPACE, ProgChunks, ProgIter,
                                     ProgPartial, ProgressIter,
                                     VALID_PROGRESS_TYPES, get_num_chunks,
                                     log_progress, progiter, progress_str,
                                     test_progress,)
    from utool.util_profile import (Profiler, cStringIO,
                                    clean_line_profile_text, clean_lprof_file,
                                    dump_profile_text, fix_rawprofile_blocks,
                                    get_block_id, get_block_totaltime,
                                    get_profile_text, get_summary,
                                    make_profiler, parse_rawprofile_blocks,
                                    parse_timemap_from_blocks,)
    from utool.util_project import (GrepResult, SetupRepo, UserProfile,
                                    ensure_text, ensure_user_profile,
                                    glob_projects, grep_projects,
                                    ibeis_user_profile, sed_projects,
                                    setup_repo,)
    from utool.util_parallel import (KillableProcess, KillableThread, bgfunc,
                                     buffered_generator, generate2,
                                     get_default_numprocs,
                                     get_sys_thread_limit, in_main_process,
                                     init_worker, set_num_procs,
                                     spawn_background_daemon_thread,
                                     spawn_background_process,
                                     spawn_background_thread,)
    from utool.util_resources import (available_memory, current_memory_usage,
                                      get_matching_process_ids,
                                      get_memstats_str,
                                      get_python_datastructure_sizes,
                                      get_resource_limits,
                                      get_resource_usage_str, memstats,
                                      num_cpus, num_unused_cpus, peak_memory,
                                      print_resource_usage, time_in_systemmode,
                                      time_in_usermode, time_str2,
                                      total_memory, used_memory,)
    from utool.util_str import (BACKSLASH, DOUBLE_QUOTE, ENABLE_COLORS,
                                NEWLINE, NO_TRUNCATE, SINGLE_QUOTE, TAUFMTSTR,
                                TAUSTR, TRIPLE_DOUBLE_QUOTE,
                                TRIPLE_SINGLE_QUOTE, align, align_lines,
                                array2string2, array_repr2, autoformat_pep8,
                                autopep8_format, bbox_str, bubbletext,
                                byte_str, byte_str2, chr_range, closet_words,
                                codeblock, color_diff_text, color_text,
                                conj_phrase, countdown_flag, dict_itemstr_list,
                                dict_str, difftext, doctest_code_line,
                                doctest_repr, ensure_ascii, ensure_unicode,
                                ensure_unicode_strlist, file_megabytes_str,
                                filesize_str, filtered_infostr, find_block_end,
                                flatten_textlines, format_multi_paragraphs,
                                format_multiple_paragraph_sentences,
                                format_single_paragraph_sentences,
                                format_text_as_docstr, func_callsig,
                                func_defsig, func_str, get_bytes,
                                get_callable_name, get_indentation,
                                get_itemstr_list, get_minimum_indentation,
                                get_textdiff, highlight_code,
                                highlight_multi_regex, highlight_regex,
                                highlight_text, horiz_string, hz_str, indent,
                                indent_list, indent_rest, indentjoin,
                                insert_block_between_lines,
                                is_byte_encoded_unicode, is_url, joins,
                                list_str, list_str_summarized,
                                long_fname_format, lorium_ipsum, msgblock,
                                number_text_lines, numpy_str,
                                order_of_magnitude_str, pack_into, packstr,
                                packtext, parse_bytes, pluralize, quantstr,
                                regex_reconstruct_split, remove_chars,
                                remove_doublenewlines, remove_doublspaces,
                                replace_between_tags, repr2, repr2_json, repr3,
                                repr4, reprfunc, scalar_str, second_str,
                                seconds_str, split_sentences2, str_between,
                                strip_ansi, testdata_text, textblock,
                                theta_str, to_camel_case, to_title_caps,
                                to_underscore_case, toggle_comment_lines,
                                trunc_repr, truncate_str,
                                unformat_text_as_docstr, unindent, utf8_len,
                                varinfo_str, verts_str,)
    from utool.util_sysreq import (ensure_in_pythonpath,
                                   get_global_dist_packages_dir,
                                   get_local_dist_packages_dir,
                                   get_site_packages_dir, in_virtual_env,
                                   is_running_as_root, locate_path,
                                   total_purge_developed_repo,)
    from utool.util_sqlite import (SQLColumnRichInfo,
                                   get_nonprimary_columninfo,
                                   get_primary_columninfo, get_table_column,
                                   get_table_columninfo_list,
                                   get_table_columnname_list,
                                   get_table_columns, get_table_csv,
                                   get_table_num_rows, get_table_rows,
                                   get_tablenames, print_database_structure,)
    from utool.util_setup import (NOOP, SETUP_PATTERNS, SetupManager,
                                  assert_in_setup_repo, autogen_sphinx_apidoc,
                                  build_pyo, clean, find_ext_modules,
                                  find_packages, get_cmdclass,
                                  get_numpy_include_dir, parse_author,
                                  parse_package_for_version, parse_readme,
                                  presetup, presetup_commands, read_license,
                                  setup_chmod, setuptools_setup,)
    from utool.util_set import (OrderedSet, oset,)
    from utool.util_regex import (REGEX_C_COMMENT, REGEX_ESCSTR, REGEX_FLOAT,
                                  REGEX_INT, REGEX_LATEX_COMMENT,
                                  REGEX_NONGREEDY, REGEX_RVAL, REGEX_STR,
                                  REGEX_VARNAME, REGEX_WHITESPACE, RE_FLAGS,
                                  RE_KWARGS, backref_field, bref_field,
                                  convert_text_to_varname, extend_regex2,
                                  extend_regex3, get_match_text,
                                  modify_quoted_strs, named_field,
                                  named_field_regex, named_field_repl,
                                  negative_lookahead, negative_lookbehind,
                                  nongreedy_kleene_star, padded_parse,
                                  parse_docblock, parse_python_syntax,
                                  positive_lookahead, positive_lookbehind,
                                  regex_get_match, regex_matches, regex_or,
                                  regex_parse, regex_replace,
                                  regex_replace_lines, regex_search,
                                  regex_split, regex_word, whole_word,)
    from utool.util_time import (Timer, Timerit, date_to_datetime,
                                 datetime_to_posixtime,
                                 determine_timestamp_format, ensure_timedelta,
                                 exiftime_to_unixtime, get_datestamp,
                                 get_posix_timedelta_str,
                                 get_posix_timedelta_str2, get_timedelta_str,
                                 get_timestamp, get_timestats_dict,
                                 get_timestats_str, get_unix_timedelta,
                                 get_unix_timedelta_str, local_timezone,
                                 parse_timedelta_str, parse_timestamp, tic,
                                 timestamp, toc, unixtime_to_datetimeobj,
                                 unixtime_to_datetimestr,
                                 unixtime_to_timedelta, utcnow_tz,)
    from utool.util_tags import (alias_tags, build_alias_map,
                                 filterflags_general_tags, modify_tags,
                                 tag_coocurrence, tag_hist,)
    from utool.util_type import (BooleanType, COMPARABLE_TYPES, FloatType,
                                 HAVE_PANDAS, IntType, LISTLIKE_TYPES,
                                 LongType, NUMPY_SCALAR_NAMES,
                                 NUMPY_TYPE_TUPLE, PRIMATIVE_TYPES,
                                 VALID_BOOL_TYPES, VALID_FLOAT_TYPES,
                                 VALID_INT_TYPES, VALID_STRING_TYPES,
                                 assert_int, bool_from_str, float_type_,
                                 fuzzy_int, fuzzy_subset,
                                 get_homogenous_list_type, get_type, int_type_,
                                 is_bool, is_comparable_type, is_dict,
                                 is_float, is_func_or_method,
                                 is_func_or_method_or_partial, is_funclike,
                                 is_int, is_list, is_listlike, is_method,
                                 is_str, is_tuple, is_type, is_valid_floattype,
                                 smart_cast, smart_cast2, try_cast, type_str,)
    from utool.util_tests import (BIGFACE, DEBUG_SRC, ExitTestException,
                                  HAPPY_FACE, HAPPY_FACE_BIG, HAPPY_FACE_SMALL,
                                  INDENT_TEST, ModuleDoctestTup, PRINT_FACE,
                                  PRINT_SRC, SAD_FACE, SAD_FACE_BIG,
                                  SAD_FACE_SMALL, SYSEXIT_ON_FAIL, TestTuple,
                                  VERBOSE_TEST, VERBOSE_TIMER, doctest_funcs,
                                  doctest_module_list, doctest_was_requested,
                                  execute_doctest, find_doctestable_modnames,
                                  find_testfunc, find_untested_modpaths,
                                  get_doctest_examples, get_module_completions,
                                  get_module_doctest_tup, get_module_testlines,
                                  get_package_testables, main_function_tester,
                                  parse_docblocks_from_docstr,
                                  parse_doctest_from_docstr, qt4ensure,
                                  qtensure, quit_if_noshow, read_exampleblock,
                                  run_test, show_if_requested,
                                  show_was_requested,)
    from utool.util_web import (find_open_port, get_localhost,
                                is_local_port_open, render_html,
                                start_simple_webserver,)
    from utool.DynamicStruct import (DynStruct,)
    from utool.Preferences import (Pref, PrefChoice, PrefInternal, PrefNode,
                                   PrefTree, VERBOSE_PREF, test_Preferences,)
    print, rrr, profile = util_inject.inject2(__name__, '[utool]')


    def reassign_submodule_attributes(verbose=1):
        """
        Updates attributes in the __init__ modules with updated attributes
        in the submodules.
        """
        import sys
        if verbose and '--quiet' not in sys.argv:
            print('dev reimport')
        # Self import
        import utool
        # Implicit reassignment.
        seen_ = set([])
        for tup in IMPORT_TUPLES:
            if len(tup) > 2 and tup[2]:
                continue  # dont import package names
            submodname, fromimports = tup[0:2]
            submod = getattr(utool, submodname)
            for attr in dir(submod):
                if attr.startswith('_'):
                    continue
                if attr in seen_:
                    # This just holds off bad behavior
                    # but it does mimic normal util_import behavior
                    # which is good
                    continue
                seen_.add(attr)
                setattr(utool, attr, getattr(submod, attr))

    def reload_subs(verbose=1):
        """ Reloads utool and submodules """
        if verbose:
            print('Reloading utool submodules')
        rrr(verbose > 1)
        def wrap_fbrrr(mod):
            def fbrrr(*args, **kwargs):
                """ fallback reload """
                if verbose > 0:
                    print('Auto-reload (using rrr) not setup for mod=%r' % (mod,))
            return fbrrr
        def get_rrr(mod):
            if hasattr(mod, 'rrr'):
                return mod.rrr
            else:
                return wrap_fbrrr(mod)
        def get_reload_subs(mod):
            return getattr(mod, 'reload_subs', wrap_fbrrr(mod))
        get_rrr(_internal)(verbose > 1)
        get_rrr(util_alg)(verbose > 1)
        get_rrr(util_aliases)(verbose > 1)
        get_rrr(util_arg)(verbose > 1)
        get_rrr(util_assert)(verbose > 1)
        get_rrr(util_autogen)(verbose > 1)
        get_rrr(util_cache)(verbose > 1)
        get_rrr(util_cplat)(verbose > 1)
        get_rrr(util_class)(verbose > 1)
        get_rrr(util_const)(verbose > 1)
        get_rrr(util_csv)(verbose > 1)
        get_rrr(util_config)(verbose > 1)
        get_rrr(util_dbg)(verbose > 1)
        get_rrr(util_dev)(verbose > 1)
        get_rrr(util_decor)(verbose > 1)
        get_rrr(util_dict)(verbose > 1)
        get_rrr(util_func)(verbose > 1)
        get_rrr(util_grabdata)(verbose > 1)
        get_rrr(util_gridsearch)(verbose > 1)
        get_rrr(util_git)(verbose > 1)
        get_rrr(util_latex)(verbose > 1)
        get_rrr(util_graph)(verbose > 1)
        get_rrr(util_hash)(verbose > 1)
        get_rrr(util_import)(verbose > 1)
        get_rrr(util_inject)(verbose > 1)
        get_rrr(util_io)(verbose > 1)
        get_rrr(util_iter)(verbose > 1)
        get_rrr(util_inspect)(verbose > 1)
        get_rrr(util_ipynb)(verbose > 1)
        get_rrr(util_logging)(verbose > 1)
        get_rrr(util_list)(verbose > 1)
        get_rrr(util_num)(verbose > 1)
        get_rrr(util_numpy)(verbose > 1)
        get_rrr(util_path)(verbose > 1)
        get_rrr(util_print)(verbose > 1)
        get_rrr(util_progress)(verbose > 1)
        get_rrr(util_profile)(verbose > 1)
        get_rrr(util_project)(verbose > 1)
        get_rrr(util_parallel)(verbose > 1)
        get_rrr(util_resources)(verbose > 1)
        get_rrr(util_str)(verbose > 1)
        get_rrr(util_sysreq)(verbose > 1)
        get_rrr(util_sqlite)(verbose > 1)
        get_rrr(util_setup)(verbose > 1)
        get_rrr(util_set)(verbose > 1)
        get_rrr(util_regex)(verbose > 1)
        get_rrr(util_time)(verbose > 1)
        get_rrr(util_tags)(verbose > 1)
        get_rrr(util_type)(verbose > 1)
        get_rrr(util_tests)(verbose > 1)
        get_rrr(util_web)(verbose > 1)
        get_rrr(DynamicStruct)(verbose > 1)
        get_rrr(Preferences)(verbose > 1)
        rrr(verbose > 1)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)
    rrrr = reload_subs
    # </AUTOGEN_INIT>
