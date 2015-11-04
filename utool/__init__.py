# -*- coding: utf-8 -*-
"""
UTool - Useful Utility Tools
   Your friendly neighborhood utility tools

TODO: INSERT APACHE VERSION 2.0 LICENCE IN ALL FILES
(Although it should be implied that the entire module and repo is released under
that licence.)
"""
# Utool is released under the Apache License Version 2.0

# flake8: noqa
# We hope to support python3
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import textwrap

#__fun_version__ = '(.878 + .478i)'
#__version__ = '1.0.0.dev1'
#__version__ = '1.1.0.dev1'
__version__ = '1.1.2.dev1'

__DYNAMIC__ = True
if __DYNAMIC__:
    __DYNAMIC__ = not '--nodyn' in sys.argv
else:
    __DYNAMIC__ = '--dyn' in sys.argv
# THESE COMMANDS WILL WRITE THE IMPORT FILE
"""
python -c "import utool" --dump-utool-init
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
    ('util_alg',       ['cartesian', 'almost_eq',]),
    ('util_aliases',   ['ddict' ,'odict']),
    ('util_arg',       ['get_argval', 'get_argflag', 'argv_flag_dec', 'QUIET',
                        'VERBOSE']),
    ('util_assert',    None),
    ('util_autogen',   None),
    ('util_cache',     ['global_cache_read', 'global_cache_write']),
    ('util_cplat',     ['cmd', 'view_directory',]),
    ('util_class',     None),
    ('util_csv',       None),
    ('util_config',    None),
    ('util_dbg',       ['execstr_dict', 'save_testdata', 'load_testdata',
                        'get_caller_name', 'import_testdata', 'embed',
                        'quitflag', 'inIPython', 'printvar2', 'all_rrr']),
    ('util_dev',       ['get_stats_str', 'get_stats', 'myprint',
                        'get_object_size']),
    ('util_decor',     ['ignores_exc_tb', 'indent_func', 'accepts_numpy',
                        'accepts_scalar_input_vector_output',
                        'accepts_scalar_input']),
    ('util_distances', ['nearest_point',]),
    ('util_dict',      None),
    ('util_func',      None),
    ('util_grabdata',  None),
    ('util_gridsearch',  None),
    ('util_git',       None),
    ('util_latex',     None),
    ('util_hash',      ['hashstr_arr', 'hashstr',]),
    ('util_import',    None),
    ('util_inject',    ['inject', 'inject_all', 'inject_print_functions']),
    ('util_io',        None),
    ('util_iter',      ['iflatten', 'ichunks', 'interleave',]),
    ('util_inspect',   None),
    ('util_logging',   None),
    ('util_list',      ['alloc_lists', 'list_index', 'npfind', 'index_of',
                        'flatten']),
    ('util_num',       None),
    ('util_numpy',       None),
    ('util_path',      ['checkpath', 'ensuredir', 'assertpath', 'truepath',
                        'list_images', 'copy', 'glob', 'grep']),
    ('util_print',     ['horiz_print', 'printshape', 'Indenter']),
    ('util_progress',  ['progress_func']),
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
    ('util_type',      None),
    ('util_tests',     None),
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
    from utool import util_csv
    from utool import util_config
    from utool import util_dbg
    from utool import util_dev
    from utool import util_decor
    from utool import util_distances
    from utool import util_dict
    from utool import util_func
    from utool import util_grabdata
    from utool import util_gridsearch
    from utool import util_git
    from utool import util_latex
    from utool import util_hash
    from utool import util_import
    from utool import util_inject
    from utool import util_io
    from utool import util_iter
    from utool import util_inspect
    from utool import util_logging
    from utool import util_list
    from utool import util_num
    from utool import util_numpy
    from utool import util_path
    from utool import util_print
    from utool import util_progress
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
    from utool import util_type
    from utool import util_tests
    from utool import DynamicStruct
    from utool import Preferences
     
    from utool.util_alg import (HAVE_NUMPY, HAVE_SCIPY, PHI, PHI_A, PHI_B, 
                                absdiff, almost_eq, bayes_rule, cartesian, 
                                choose, cumsum, deg_to_rad, enumerate_primes, 
                                euclidean_dist, fibonacci, fibonacci_iterative, 
                                fibonacci_recursive, find_std_inliers, 
                                flatten_membership_mapping, generate_primes, 
                                get_nth_prime, get_nth_prime_bruteforce, 
                                get_phi, get_phi_ratio1, get_prime_index, 
                                greedy_max_inden_setcover, haversine, inbounds, 
                                is_prime, item_hist, knapsack, knapsack_greedy, 
                                max_size_max_distance_subset, 
                                maximin_distance_subset1d, 
                                maximum_distance_subset, 
                                negative_minclamp_inplace, norm_zero_one, 
                                normalize, rad_to_deg, safe_div, safe_max, 
                                safe_min, safe_pdist, search_utool, 
                                triangular_number, unixtime_hourdiff, 
                                xywh_to_tlbr,) 
    from utool.util_aliases import (OrderedDict, ddict, defaultdict, iprod, 
                                    namedtuple, odict, product,) 
    from utool.util_arg import (ArgumentParser2, DEBUG2, NOT_QUIET, NO_ASSERTS, 
                                QUIET, REPORT, SAFE, SILENT, STRICT, 
                                SUPER_STRICT, TRACE, USE_ASSERT, VERBOSE, 
                                VERBOSE_ARGPARSE, VERYVERBOSE, argparse_dict, 
                                argv_flag_dec, argv_flag_dec_true, 
                                autogen_argparse2, autogen_argparse_block, 
                                get_arg, get_arg_dict, get_argflag, 
                                get_argv_tail, get_argval, 
                                get_dict_vals_from_commandline, get_flag, 
                                get_fpath_args, get_module_verbosity_flags, 
                                get_verbflag, make_argparse2, 
                                parse_arglist_hack, parse_cfgstr_list, 
                                parse_dict_from_argv, reset_argrecord, 
                                switch_sanataize,) 
    from utool.util_assert import (assert_all_eq, assert_all_in, 
                                   assert_all_not_None, assert_almost_eq, 
                                   assert_eq, assert_inbounds, assert_lessthan, 
                                   assert_lists_eq, assert_same_len, 
                                   assert_scalar_list, assert_unflat_level, 
                                   get_first_None_position, lists_eq,) 
    from utool.util_autogen import (PythonStatement, auto_docstr, 
                                    autofix_codeblock, deque, 
                                    find_modname_in_pythonpath, 
                                    is_modname_in_pythonpath, 
                                    load_func_from_module, make_args_docstr, 
                                    make_cmdline_docstr, make_default_docstr, 
                                    make_default_module_maintest, 
                                    make_docstr_block, make_example_docstr, 
                                    make_returns_or_yeilds_docstr, 
                                    print_auto_docstr, write_modscript_alias,) 
    from utool.util_cache import (BadZipFile, Cachable, CacheMissException, 
                                  Cacher, GlobalShelfContext, LRUDict, 
                                  LazyDict, LazyList, ShelfCacher, USE_CACHE, 
                                  UtoolJSONEncoder, VERBOSE_CACHE, cached_func, 
                                  cachestr_repr, chain, consensed_cfgstr, 
                                  delete_global_cache, from_json, 
                                  get_cfgstr_from_args, get_default_appname, 
                                  get_func_result_cachekey, 
                                  get_global_cache_dir, get_global_shelf_fpath, 
                                  get_lru_cache, global_cache_dump, 
                                  global_cache_read, global_cache_write, 
                                  load_cache, save_cache, shelf_open, 
                                  text_dict_read, text_dict_write, 
                                  time_different_diskstores, to_json, 
                                  tryload_cache, tryload_cache_list, 
                                  tryload_cache_list_with_compute, 
                                  view_global_cache_dir,) 
    from utool.util_cplat import (COMPUTER_NAME, DARWIN, LIB_DICT, 
                                  LIB_EXT_LIST, LINUX, OS_TYPE, PYLIB_DICT, 
                                  WIN32, assert_installed_debian, 
                                  change_term_title, check_installed_debian, 
                                  chmod_add_executable, cmd, editfile, 
                                  ensure_app_resource_dir, 
                                  get_app_resource_dir, get_computer_name, 
                                  get_dir_diskspaces, get_disk_space, 
                                  get_dynamic_lib_globstrs, 
                                  get_dynlib_dependencies, get_dynlib_exports, 
                                  get_file_info, get_file_nBytes, 
                                  get_file_nBytes_str, get_flops, 
                                  get_free_diskbytes, get_install_dirs, 
                                  get_lib_ext, get_path_dirs, get_pylib_ext, 
                                  get_python_dynlib, get_resource_dir, 
                                  get_user_name, geteditor, getroot, 
                                  in_pyinstaller_package, inject, 
                                  ipython_paste, is64bit_python, 
                                  is_file_executable, is_file_writable, 
                                  ls_libs, print_dir_diskspace, print_path, 
                                  print_system_users, python_executable, 
                                  quote_single_command, run_realtime_process, 
                                  search_env_paths, send_keyboard_input, 
                                  set_process_title, shell, 
                                  spawn_delayed_ipython_paste, startfile, 
                                  truepath, unixpath, unload_module, vd, 
                                  view_directory,) 
    from utool.util_class import (KwargsWrapper, QUIET_CLASS, 
                                  ReloadingMetaclass, VERBOSE_CLASS, 
                                  autogen_explicit_injectable_metaclass, 
                                  autogen_import_list, decorate_class_method, 
                                  decorate_postinject, get_comparison_methods, 
                                  get_funcname, get_injected_modules, 
                                  inject_all_external_modules, 
                                  inject_func_as_method, inject_instance, 
                                  makeForwardingMetaclass, 
                                  make_class_method_decorator, 
                                  make_class_postinject_decorator, 
                                  postinject_instance, private_rrr_factory, 
                                  reload_class_methods, 
                                  reload_injected_modules, 
                                  reloading_meta_metaclass_factory, 
                                  remove_private_obfuscation, 
                                  test_reloading_metaclass,) 
    from utool.util_csv import (make_csv_table, numpy_to_csv,) 
    from utool.util_config import (get_default_global_config, 
                                   get_default_repo_config, read_repo_config, 
                                   write_default_repo_config,) 
    from utool.util_dbg import (COLORED_EXCEPTIONS, EmbedOnException, FORCE_TB, 
                                IPYTHON_EMBED_STR, RAISE_ALL, TB, all_rrr, 
                                breakpoint, debug_exception, debug_hstack, 
                                debug_list, debug_npstack, debug_vstack, 
                                dict_dbgstr, embed, embed2, 
                                embed_on_exception_context, eoxc, 
                                execstr_attr_list, execstr_dict, execstr_embed, 
                                execstr_func, execstr_parent_locals, 
                                execstr_src, explore_module, explore_stack, 
                                fmtlocals, formatex, get_caller_lineno, 
                                get_caller_locals, get_caller_modname, 
                                get_caller_name, get_caller_prefix, 
                                get_caller_stack_frame, 
                                get_current_stack_depth, 
                                get_localvar_from_stack, get_parent_frame, 
                                get_parent_globals, get_parent_locals, 
                                get_reprs, get_stack_frame, get_var_from_stack, 
                                get_varname_from_locals, 
                                get_varname_from_stack, get_varstr, 
                                get_varval_from_locals, haveIPython, 
                                import_testdata, inIPython, 
                                in_jupyter_notebook, ipython_execstr, 
                                ipython_execstr2, is_valid_varname, 
                                keys_dbgstr, len_dbgstr, list_dbgstr, 
                                load_testdata, module_functions, 
                                parse_locals_keylist, print_frame, print_keys, 
                                print_traceback, print_varlen, printex, 
                                printvar, printvar2, public_attributes, qflag, 
                                quasiquote, quit, quitflag, save_testdata, 
                                search_stack_for_localvar, 
                                search_stack_for_var, split, super_print, 
                                varname_regex,) 
    from utool.util_dev import (ClassNoParam, DEPRICATED, DEVELOPER_MODE, 
                                ENABLE_MEMTRACK, INDEXABLE_TYPES, 
                                InteractiveIter, MemoryTracker, NoParam, 
                                STAT_KEY_ORDER, USER_MODE, are_you_sure, 
                                autopep8_diff, compile_cython, 
                                copy_text_to_clipboard, 
                                disable_garbage_collection, 
                                enable_garbage_collection, ensure_str_list, 
                                extract_timeit_setup, find_exe, 
                                find_interesting_stats, garbage_collect, 
                                get_clipboard, get_cython_exe, 
                                get_jagged_stats, get_nonconflicting_path, 
                                get_nonconflicting_path_old, 
                                get_nonconflicting_string, get_object_base, 
                                get_object_size, get_object_size_str, 
                                get_partial_func_name, get_statdict, get_stats, 
                                get_stats_str, info, init_catch_ctrl_c, 
                                input_timeout, inverable_group_multi_list, 
                                inverable_unique_two_lists, is_developer, iup, 
                                listinfo, make_at_least_n_items_valid, 
                                make_call_graph, make_object_graph, 
                                memory_dump, myprint, npArrInfo, npinfo, 
                                numpy_list_num_bits, print_object_size, 
                                print_object_size_tree, print_stats, 
                                printableType, printableVal, report_memsize, 
                                reset_catch_ctrl_c, set_clipboard, 
                                strip_line_comments, testit, timeit_compare, 
                                timeit_grid, tuples_to_unique_scalars, 
                                uninvert_unique_two_lists, 
                                user_cmdline_prompt,) 
    from utool.util_decor import (IGNORE_TRACEBACK, NOINDENT_DECOR, 
                                  ONEX_REPORT_INPUT, PROFILING, SIG_PRESERVE, 
                                  UNIQUE_NUMPY, accepts_numpy, 
                                  accepts_scalar_input, accepts_scalar_input2, 
                                  accepts_scalar_input_vector_output, 
                                  apply_docstr, dummy_args_decor, getter_1to1, 
                                  getter_1toM, ignores_exc_tb, indent_func, 
                                  interested, lazyfunc, memoize, 
                                  memoize_nonzero, memoize_single, 
                                  memoize_zero, on_exception_report_input, 
                                  preserve_sig, show_return_value, 
                                  test_ignore_exec_traceback, time_func, 
                                  tracefunc, tracefunc_xml,) 
    from utool.util_distances import (nearest_point,) 
    from utool.util_dict import (AutoVivification, all_dict_combinations, 
                                 all_dict_combinations_lbls, 
                                 all_dict_combinations_ordered, 
                                 assert_keys_are_subset, augdict, 
                                 build_conflict_dict, count_dict_vals, 
                                 delete_dict_keys, delete_keys, dict_assign, 
                                 dict_filter_nones, dict_find_keys, 
                                 dict_find_other_sameval_keys, dict_hist, 
                                 dict_hist_cumsum, dict_intersection, 
                                 dict_isect, dict_keysubset, 
                                 dict_map_apply_vals, dict_setdiff, dict_stack, 
                                 dict_subset, dict_take, dict_take_gen, 
                                 dict_take_list, dict_take_pop, 
                                 dict_to_keyvals, dict_union, dict_union2, 
                                 dict_union3, dict_update_newkeys, 
                                 dict_val_map, dict_where_len0, dictinfo, 
                                 get_dict_column, get_dict_hashid, group_items, 
                                 groupby_tags, hierarchical_group_items, 
                                 hierarchical_map_vals, hmap_vals, 
                                 iflatten_dict_values, invert_dict, is_dicteq, 
                                 items_sorted_by_value, 
                                 iter_all_dict_combinations_ordered, 
                                 iteritems_sorted, keys_sorted_by_value, 
                                 map_dict_keys, map_dict_vals, merge_dicts, 
                                 move_odict_item, order_dict_by, partial, 
                                 update_existing, updateif_haskey,) 
    from utool.util_func import (compose_functions, general_get, general_set, 
                                 identity, uinput_1to1,) 
    from utool.util_grabdata import (BadZipfile, TESTIMG_URL_DICT, 
                                     archive_files, clear_test_img_cache, 
                                     download_url, 
                                     experiment_download_multiple_urls, 
                                     fix_dropbox_link, geo_locate, 
                                     get_prefered_browser, 
                                     get_valid_test_imgkeys, grab_file_url, 
                                     grab_selenium_chromedriver, 
                                     grab_selenium_driver, grab_test_imgpath, 
                                     grab_zipped_url, open_url_in_browser, 
                                     scp_pull, split_archive_ext, 
                                     unarchive_file, untar_file, unzip_file, 
                                     url_read,) 
    from utool.util_gridsearch import (DimensionBasis, GridSearch, ParamInfo, 
                                       ParamInfoBool, ParamInfoList, 
                                       constrain_cfgdict_list, 
                                       get_cfgdict_lbl_list_subset, 
                                       get_cfgdict_list_subset, 
                                       grid_search_generator, gridsearch_timer, 
                                       interact_gridsearch_result_images, 
                                       make_cfglbls, 
                                       make_constrained_cfg_and_lbl_list, 
                                       testdata_grid_search,) 
    from utool.util_git import (CODE_DIR, DRY_RUN, PROJECT_REPO_DIRS, 
                                PROJECT_REPO_URLS, checkout_repos, 
                                ensure_project_repos, ensure_repos, 
                                get_project_repo_dirs, get_repo_dirs, 
                                get_repo_dname, gg_command, gitcmd, is_gitrepo, 
                                isdir, pull_repos, rename_branch, repo_list, 
                                set_code_dir, set_project_repos, set_userid, 
                                setup_develop_repos, std_build_command,) 
    from utool.util_latex import (compile_latex_text, compress_pdf, 
                                  ensure_colvec, ensure_rowvec, escape_latex, 
                                  find_ghostscript_exe, get_bibtex_dict, 
                                  get_latex_figure_str, get_latex_figure_str2, 
                                  is_substr, latex_get_stats, 
                                  latex_multicolumn, latex_multirow, 
                                  latex_newcommand, 
                                  latex_sanatize_command_name, latex_scalar, 
                                  long_substr, make_full_document, 
                                  make_score_tabular, make_stats_tabular, 
                                  render, render_latex_text, replace_all, 
                                  tabular_join,) 
    from utool.util_hash import (ALPHABET, ALPHABET_16, ALPHABET_27, BIGBASE, 
                                 DictProxyType, HASH_LEN, augment_uuid, 
                                 convert_hexstr_to_bigbase, deterministic_uuid, 
                                 get_file_hash, get_file_uuid, get_zero_uuid, 
                                 hashable_to_uuid, hashstr, hashstr27, 
                                 hashstr_arr, hashstr_arr27, hashstr_md5, 
                                 hashstr_sha1, image_uuid, make_hash, 
                                 random_nonce, random_uuid,) 
    from utool.util_import import (LazyModule, import_modname, 
                                   import_module_from_fpath, lazy_module_attrs, 
                                   package_contents, tryimport,) 
    from utool.util_inject import (ARGV_DEBUG_FLAGS, DEBUG_PRINT, 
                                   DEBUG_PRINT_N, DUMMYPROF_FUNC, 
                                   HAVE_PYGMENTS, KERNPROF_FUNC, 
                                   PRINT_INJECT_ORDER, PROF_FUNC_PAT_LIST, 
                                   PROF_MOD_PAT_LIST, TIMERPROF_FUNC, argv, 
                                   colored_pygments_excepthook, inject, 
                                   inject2, inject_all, 
                                   inject_colored_exceptions, 
                                   inject_print_functions, 
                                   inject_profile_function, inject_python_code, 
                                   inject_python_code2, inject_reload_function, 
                                   memprof, noinject, 
                                   split_python_text_into_lines, wraps,) 
    from utool.util_io import (HAS_NUMPY, HAVE_LOCKFILE, load_cPkl, load_data, 
                               load_hdf5, load_numpy, load_pytables, load_text, 
                               lock_and_load_cPkl, lock_and_save_cPkl, 
                               read_from, readfrom, save_cPkl, save_data, 
                               save_hdf5, save_numpy, save_pytables, save_text, 
                               try_decode, write_to, writeto,) 
    from utool.util_iter import (and_iters, cycle, ensure_iterable, 
                                 evaluate_generator, ichunks, ichunks_list, 
                                 ifilter_Nones, ifilter_items, 
                                 ifilterfalse_items, iflatten, 
                                 iflatten_scalars, iget_list_column, 
                                 iget_list_column_slice, interleave, 
                                 interleave2, interleave3, isiterable, islice, 
                                 iter_multichunks, iter_window, itertwo, 
                                 next_counter, roundrobin, wrap_iterable,) 
    from utool.util_inspect import (KWReg, dummy_func, exec_func_sourcecode, 
                                    exec_func_src, filter_valid_kwargs, 
                                    find_child_kwarg_funcs, 
                                    find_pattern_above_row, 
                                    find_pyclass_above_row, 
                                    find_pyfunc_above_row, get_argnames, 
                                    get_dev_hints, get_docstr, 
                                    get_func_argspec, get_func_kwargs, 
                                    get_func_sourcecode, get_funcdoc, 
                                    get_funcfpath, get_funcglobals, get_imfunc, 
                                    get_kwargs, get_kwdefaults, 
                                    get_kwdefaults2, 
                                    get_module_owned_functions, 
                                    infer_arg_types_and_descriptions, 
                                    infer_function_info, inherit_kwargs, 
                                    iter_module_doctestable, 
                                    list_class_funcnames, 
                                    list_global_funcnames, parse_callname, 
                                    parse_func_kwarg_keys, parse_kwarg_keys, 
                                    parse_return_type, prettyprint_parsetree, 
                                    recursive_parse_kwargs, set_funcdoc, 
                                    set_funcname,) 
    from utool.util_logging import (LOGGING_VERBOSE, PRINT_ALL_CALLERS, 
                                    add_logging_handler, get_current_log_fpath, 
                                    get_current_log_text, get_log_fpath, 
                                    get_logging_dir, get_utool_logger, 
                                    logdir_cacheid, start_logging, 
                                    stop_logging,) 
    from utool.util_list import (accumulate, alloc_lists, alloc_nones, 
                                 and_lists, debug_consec_list, 
                                 debug_duplicate_items, delete_items_by_index, 
                                 depth_profile, duplicates_exist, 
                                 ensure_list_size, filter_Nones, filter_items, 
                                 filter_startswith, filterfalse_items, 
                                 find_duplicate_items, find_list_indexes, 
                                 find_nonconsec_indices, flag_None_items, 
                                 flag_unique_items, flatten, flattenize, 
                                 get_callable_name, get_dirty_items, 
                                 get_list_column, get_list_column_slice, 
                                 group_consecutives, group_consecutives_numpy, 
                                 iflag_unique_items, intersect_ordered, 
                                 invertible_flatten, invertible_flatten2, 
                                 invertible_flatten2_numpy, is_subset_of_any, 
                                 issorted, isunique, list_all_eq_to, 
                                 list_allsame, list_argmax, list_argsort, 
                                 list_compress, list_compresstake, list_cover, 
                                 list_deep_types, list_depth, 
                                 list_intersection, list_inverse_take, 
                                 list_issubset, list_issuperset, list_replace, 
                                 list_roll, list_set_equal, list_take, 
                                 list_transpose, list_type_profile, 
                                 list_unflat_take, list_where, 
                                 list_zipcompress, list_zipflatten, 
                                 list_ziptake, listclip, listfind, lmap, 
                                 make_index_lookup, make_sortby_func, 
                                 multi_replace, not_list, or_lists, 
                                 partial_imap_1to1, print_duplicate_map, 
                                 priority_argsort, priority_sort, 
                                 recursive_replace, replace_nones, 
                                 safe_listget, safe_slice, sample_lists, 
                                 sample_zip, scalar_input_map, search_list, 
                                 setdiff_ordered, setintersect_ordered, 
                                 sortedby, sortedby2, strided_sample, tuplize, 
                                 unflat_unique_rowid_map, unflatten, 
                                 unflatten2, unique_keep_order, 
                                 unique_ordered, unique_unordered, xor_lists,) 
    from utool.util_num import (commas, fewest_digits_float_str, 
                                float_to_decimal, format_, get_sys_maxfloat, 
                                get_sys_maxint, get_sys_minint, int_comma_str, 
                                num2_sigfig, num_fmt, order_of_magnitude_ceil, 
                                sigfig_str,) 
    from utool.util_numpy import (deterministic_sample, deterministic_shuffle, 
                                  index_of, intersect2d, make_incrementer, 
                                  npfind, random_indexes, random_sample, 
                                  sample_domain, spaced_indexes, spaced_items, 
                                  tiled_range,) 
    from utool.util_path import (ChdirContext, IMG_EXTENSIONS, PRINT_CALLER, 
                                 append_suffixlist_to_namelist, assert_exists, 
                                 assertpath, augpath, basename_noext, 
                                 checkpath, copy, copy_all, copy_files_to, 
                                 copy_list, copy_single, copy_worker, delete, 
                                 dirsplit, ensure_crossplat_path, 
                                 ensure_mingw_drive, ensure_native_path, 
                                 ensure_unixslash, ensuredir, ensurepath, 
                                 existing_subpath, extend_regex, file_bytes, 
                                 file_megabytes, find_lib_fpath, 
                                 fixwin32_shortname, fnames_to_fpaths, 
                                 fpaths_to_fnames, get_basename_noext_list, 
                                 get_basepath_list, get_ext_list, 
                                 get_modname_from_modpath, 
                                 get_modpath_from_modname, get_module_dir, 
                                 get_module_subdir_list, get_path_type, 
                                 get_relative_modpath, 
                                 get_standard_exclude_dnames, 
                                 get_standard_include_patterns, 
                                 get_win32_short_path_name, glob, 
                                 glob_python_modules, grep, grepfile, iglob, 
                                 is_module_dir, is_private_module, isfile, 
                                 islink, ismount, list_images, 
                                 longest_existing_path, ls, ls_dirs, ls_images, 
                                 ls_moduledirs, ls_modulefiles, matches_image, 
                                 matching_fnames, move, move_list, newcd, 
                                 num_images_in_dir, path_ndir_split, 
                                 pathsplit_full, platform_path, progress_func, 
                                 relpath_unix, remove_dirs, 
                                 remove_existing_fpaths, remove_file, 
                                 remove_file_list, remove_files_in_dir, 
                                 remove_fpaths, search_candidate_paths, 
                                 search_in_dirs, splitdrive, symlink, tail, 
                                 touch, truepath, truepath_relative, unixjoin, 
                                 win_shortcut,) 
    from utool.util_print import (Indenter, NO_INDENT, PrintStartEndContext, 
                                  colorprint, dictprint, get_colored_diff, 
                                  horiz_print, printNOTQUIET, printVERBOSE, 
                                  printWARN, print_code, print_dict, 
                                  print_difftext, print_filesize, print_list, 
                                  print_locals, print_python_code, printdict, 
                                  printif, printshape,) 
    from utool.util_progress import (AGGROFLUSH, FORCE_ALL_PROGRESS, 
                                     NO_PROGRESS, PROGGRESS_BACKSPACE, 
                                     PROGRESS_FLUSH, PROGRESS_WRITE, 
                                     ProgressIter, VALID_PROGRESS_TYPES, 
                                     get_nTotalChunks, log_progress, prog_func, 
                                     progiter, progress_chunks, progress_func, 
                                     progress_str, simple_progres_func, 
                                     test_progress,) 
    from utool.util_project import (UserProfile, ensure_user_profile, 
                                    grep_projects, ibeis_user_profile, 
                                    sed_projects,) 
    from utool.util_parallel import (BACKEND, MIN_PARALLEL_TASKS, 
                                     USE_GLOBAL_POOL, VERBOSE_PARALLEL, 
                                     VERYVERBOSE_PARALLEL, buffered_generator, 
                                     close_pool, ensure_pool, generate, 
                                     get_default_numprocs, in_main_process, 
                                     init_pool, init_worker, new_pool, process, 
                                     set_num_procs, spawn_background_process, 
                                     spawn_background_thread,) 
    from utool.util_resources import (available_memory, current_memory_usage, 
                                      get_matching_process_ids, 
                                      get_memstats_str, 
                                      get_python_datastructure_sizes, 
                                      get_resource_limits, 
                                      get_resource_usage_str, memstats, 
                                      num_cpus, peak_memory, 
                                      print_resource_usage, time_in_systemmode, 
                                      time_in_usermode, time_str2, 
                                      total_memory, used_memory,) 
    from utool.util_str import (DOUBLE_QUOTE, NEWLINE, SINGLE_QUOTE, TAU, 
                                TAUFMTSTR, TAUSTR, TRIPLE_DOUBLE_QUOTE, 
                                TRIPLE_SINGLE_QUOTE, USE_GLOBAL_INFO, align, 
                                align_lines, array2string2, array_repr2, 
                                autoformat_pep8, bbox_str, bubbletext, 
                                byte_str, byte_str2, clipstr, closet_words, 
                                codeblock, conj_phrase, dict_itemstr_list, 
                                dict_str, doctest_code_line, doctest_repr, 
                                edit_distance, ensure_unicode_strlist, 
                                file_megabytes_str, filesize_str, 
                                flatten_textlines, format_text_as_docstr, 
                                func_str, get_freespace_str, get_indentation, 
                                get_itemstr_list, get_minimum_indentation, 
                                get_textdiff, get_unix_timedelta_str, 
                                horiz_string, hz_str, indent, indent_list, 
                                indent_rest, indentcat, indentjoin, 
                                is_byte_encoded_unicode, is_url, joins, 
                                list_str, list_str_summarized, listinfo_str, 
                                long_fname_format, lorium_ipsum, msgblock, 
                                newlined_list, number_text_lines, numeric_str, 
                                numpy_str, numpy_str2, obj_str, 
                                order_of_magnitude_str, pack_into, packstr, 
                                padded_str_range, percent_str, pluralize, 
                                quantity_str, remove_chars, remove_doublspaces, 
                                remove_vowels, replace_between_tags, 
                                replace_nonquoted_text, reprfunc, scalar_str, 
                                seconds_str, singular_string, str2, 
                                str_between, textblock, theta_str, 
                                to_camel_case, truncate_str, tupstr, 
                                unformat_text_as_docstr, unindent, verts_str,) 
    from utool.util_sysreq import (ensure_in_pythonpath, get_site_packages_dir, 
                                   in_virtual_env, is_running_as_root, 
                                   locate_path,) 
    from utool.util_sqlite import (SQLColumnRichInfo, 
                                   get_nonprimary_columninfo, 
                                   get_primary_columninfo, get_table_column, 
                                   get_table_columninfo_list, 
                                   get_table_columnname_list, 
                                   get_table_columns, get_table_csv, 
                                   get_table_num_rows, get_table_rows, 
                                   get_tablenames, print_database_structure,) 
    from utool.util_setup import (NOOP, SETUP_PATTERNS, assert_in_setup_repo, 
                                  autogen_sphinx_apidoc, build_pyo, clean, 
                                  find_ext_modules, find_packages, 
                                  get_cmdclass, get_numpy_include_dir, 
                                  parse_author, parse_package_for_version, 
                                  parse_readme, presetup, presetup_commands, 
                                  read_license, setup_chmod, setuptools_setup, 
                                  translate_cyth,) 
    from utool.util_set import (OrderedSet, oset,) 
    from utool.util_regex import (REGEX_ESCSTR, REGEX_FLOAT, REGEX_INT, 
                                  REGEX_NONGREEDY, REGEX_RVAL, REGEX_STR, 
                                  REGEX_VARNAME, REGEX_WHITESPACE, RE_FLAGS, 
                                  RE_KWARGS, backref_field, bref_field, 
                                  convert_text_to_varname, get_match_text, 
                                  modify_quoted_strs, named_field, 
                                  named_field_regex, named_field_repl, 
                                  negative_lookahead, negative_lookbehind, 
                                  padded_parse, parse_docblock, 
                                  parse_python_syntax, positive_lookahead, 
                                  positive_lookbehind, regex_get_match, 
                                  regex_matches, regex_or, regex_parse, 
                                  regex_replace, regex_replace_lines, 
                                  regex_search, regex_split, regex_word, sed, 
                                  sedfile, whole_word,) 
    from utool.util_time import (Timer, determine_timestamp_format, 
                                 ensure_timedelta, exiftime_to_unixtime, 
                                 get_datestamp, get_posix_timedelta_str, 
                                 get_posix_timedelta_str2, 
                                 get_printable_timestamp, get_timedelta_str, 
                                 get_timestamp, get_timestats_dict, 
                                 get_timestats_str, get_unix_timedelta, 
                                 parse_timedelta_str, tic, timestamp, toc, 
                                 unixtime_to_datetimeobj, 
                                 unixtime_to_datetimestr, 
                                 unixtime_to_timedelta,) 
    from utool.util_type import (BooleanType, FloatType, IntType, 
                                 LISTLIKE_TYPES, LongType, NP_NDARRAY, 
                                 NUMPY_SCALAR_NAMES, VALID_BOOL_TYPES, 
                                 VALID_FLOAT_TYPES, VALID_INT_TYPES, 
                                 assert_int, bool_from_str, fuzzy_int, 
                                 fuzzy_subset, get_homogenous_list_type, 
                                 get_type, is_bool, is_dict, is_float, 
                                 is_func_or_method, 
                                 is_func_or_method_or_partial, is_funclike, 
                                 is_int, is_list, is_listlike, is_method, 
                                 is_str, is_tuple, is_type, is_valid_floattype, 
                                 smart_cast, smart_cast2, try_cast, type_str,) 
    from utool.util_tests import (BIGFACE, DEBUG_SRC, EXEC_MODE, 
                                  ExitTestException, HAPPY_FACE, 
                                  HAPPY_FACE_BIG, HAPPY_FACE_SMALL, 
                                  INDENT_TEST, ModuleDoctestTup, PRINT_FACE, 
                                  PRINT_SRC, SAD_FACE, SAD_FACE_BIG, 
                                  SAD_FACE_SMALL, SYSEXIT_ON_FAIL, TestTuple, 
                                  VERBOSE_TEST, VERBOSE_TIMER, def_test, 
                                  dev_ipython_copypaster, doctest_funcs, 
                                  doctest_module_list, doctest_was_requested, 
                                  find_doctestable_modnames, 
                                  find_untested_modpaths, get_dev_paste_code, 
                                  get_doctest_examples, get_func_source, 
                                  get_module_doctest_tup, get_module_testlines, 
                                  main_function_tester, 
                                  make_run_tests_script_text, 
                                  parse_docblocks_from_docstr, 
                                  parse_doctest_from_docstr, quit_if_noshow, 
                                  run_test, show_if_requested, 
                                  show_was_requested,) 
    from utool.DynamicStruct import (DynStruct,) 
    from utool.Preferences import (Pref, PrefChoice, PrefInternal, PrefNode, 
                                   PrefTree, VERBOSE_PREF, test_Preferences,) 
    print, print_, printDBG, rrr, profile = util_inject.inject(
        __name__, '[utool]')
    
    
    def reassign_submodule_attributes(verbose=True):
        """
        why reloading all the modules doesnt do this I don't know
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
    
    
    def reload_subs(verbose=True):
        """ Reloads utool and submodules """
        rrr(verbose=verbose)
        def fbrrr(*args, **kwargs):
            """ fallback reload """
            pass
        getattr(_internal, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_alg, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_aliases, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_arg, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_assert, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_autogen, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_cache, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_cplat, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_class, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_csv, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_config, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_dbg, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_dev, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_decor, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_distances, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_dict, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_func, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_grabdata, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_gridsearch, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_git, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_latex, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_hash, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_import, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_inject, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_io, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_iter, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_inspect, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_logging, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_list, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_num, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_numpy, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_path, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_print, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_progress, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_project, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_parallel, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_resources, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_str, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_sysreq, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_sqlite, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_setup, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_set, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_regex, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_time, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_type, 'rrr', fbrrr)(verbose=verbose)
        getattr(util_tests, 'rrr', fbrrr)(verbose=verbose)
        getattr(DynamicStruct, 'rrr', fbrrr)(verbose=verbose)
        getattr(Preferences, 'rrr', fbrrr)(verbose=verbose)
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)
    rrrr = reload_subs
    # </AUTOGEN_INIT>
