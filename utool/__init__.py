"""
UTool - Useful Utility Tools
   Your friendly neighborhood utility tools

TODO: INSERT APACHE LICENCE
"""
# flake8: noqa
# We hope to support python3
from __future__ import absolute_import, division, print_function
import sys
import textwrap

#__fun_version__ = '(.878 + .478i)'
__version__ = '1.0.0.dev1'

__DYNAMIC__ = not '--nodyn' in sys.argv
#__DYNAMIC__ = '--dyn' in sys.argv
"""
python -c "import utool" --dump-utool-init
python -c "import utool" --update-utool-init
"""


def dev_reimport():
    """ why rrrr doesnt do this I dont know """
    print('dev reimport')
    import utool
    # This still doesn't even work
    utool.list_depth = utool.util_list.list_depth
    utool.make_default_docstr = utool.util_dev.make_default_docstr
    pass

if __DYNAMIC__:
    # TODO: import all utool external prereqs. Then the imports will not import
    # anything that has already in a toplevel namespace
    pass
    # COMMENTED OUT FOR FROZEN __INIT__
    # Dynamically import listed util libraries and their members.
    UTOOLS_LIST = [
        ('_internal',      None),
        ('util_alg',       ['cartesian', 'almost_eq',]),
        ('util_aliases',   ['ddict' ,'odict']),
        ('util_arg',       ['get_argval', 'get_argflag', 'argv_flag_dec', 'QUIET',
                            'VERBOSE']),
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
        ('util_git',       None),
        ('util_hash',      ['hashstr_arr', 'hashstr',]),
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
        ('util_parallel',  None),
        ('util_resources', ['print_resource_usage']),
        ('util_str',       ['byte_str2', 'horiz_string', 'theta_str']),
        ('util_sysreq',    None),
        ('util_setup',     'presetup'),
        ('util_regex',     ['regex_search']),
        ('util_time',      ['tic', 'toc', 'Timer']),
        ('util_type',      None),
        ('util_tests',     None),
        ('DynamicStruct',  ['DynStruct']),
        ('Preferences',    ['Pref']),
        ]
    from ._internal import util_importer
    # FIXME: this might actually work with rrrr, but things arent being
    # reimported because they are already in the modules list
    import_execstr = util_importer.dynamic_import(__name__, UTOOLS_LIST)
    exec(import_execstr)
else:
    # <AUTOGEN_INIT>

    from . import _internal
    from . import util_alg
    from . import util_aliases
    from . import util_arg
    from . import util_cache
    from . import util_cplat
    from . import util_class
    from . import util_csv
    from . import util_config
    from . import util_dbg
    from . import util_dev
    from . import util_decor
    from . import util_distances
    from . import util_dict
    from . import util_func
    from . import util_grabdata
    from . import util_git
    from . import util_hash
    from . import util_inject
    from . import util_io
    from . import util_iter
    from . import util_inspect
    from . import util_logging
    from . import util_list
    from . import util_num
    from . import util_numpy
    from . import util_path
    from . import util_print
    from . import util_progress
    from . import util_parallel
    from . import util_resources
    from . import util_str
    from . import util_sysreq
    from . import util_setup
    from . import util_regex
    from . import util_time
    from . import util_type
    from . import util_tests
    from . import DynamicStruct
    from . import Preferences
     
    from .util_alg import (HAS_NUMPY, PHI, PHI_A, PHI_B, almost_eq, 
                           build_reverse_mapping, cartesian, choose, 
                           defaultdict, find_std_inliers, 
                           flatten_membership_mapping, get_nth_prime, get_phi, 
                           get_phi_ratio1, group_items, iceil, iround, 
                           is_prime, norm_zero_one, normalize, 
                           unique_row_indexes, unpack_items_sorted, 
                           unpack_items_sorted_by_lenvalue, 
                           unpack_items_sorted_by_value, void_rowview_numpy, 
                           xywh_to_tlbr,) 
    from .util_aliases import (OrderedDict, ddict, iprod, odict, product,) 
    from .util_arg import (ArgumentParser2, DEBUG2, IGNORE_TRACEBACK, Indenter, 
                           NOT_QUIET, NO_ASSERTS, QUIET, REPORT, SAFE, STRICT, 
                           SUPER_STRICT, TRACE, USE_ASSERT, VERBOSE, 
                           VERYVERBOSE, argv_flag_dec, argv_flag_dec_true, 
                           fuzzy_int, get_arg, get_argflag, get_argval, 
                           get_flag, get_fpath_args, get_funcname, inject, 
                           make_argparse2, set_funcname, switch_sanataize, 
                           try_cast,) 
    from .util_cache import (Cacher, GlobalShelfContext, cached_func, chain, 
                             default_appname, delete_global_cache, 
                             get_argnames, get_cfgstr_from_args, 
                             get_default_appname, get_global_cache_dir, 
                             get_global_shelf_fpath, get_kwdefaults, 
                             global_cache_dname, global_cache_dump, 
                             global_cache_fname, global_cache_read, 
                             global_cache_write, load_cache, save_cache, 
                             text_dict_write, view_global_cache_dir,) 
    from .util_cplat import (COMPUTER_NAME, DARWIN, LIB_DICT, LIB_EXT_LIST, 
                             LINUX, OS_TYPE, PYLIB_DICT, WIN32, cmd, editfile, 
                             get_app_resource_dir, get_computer_name, 
                             get_dir_diskspaces, get_disk_space, 
                             get_dynamic_lib_globstrs, get_dynlib_dependencies, 
                             get_flops, get_free_diskbytes, get_lib_ext, 
                             get_pylib_ext, get_python_dynlib, 
                             get_resource_dir, get_user_name, getroot, 
                             is64bit_python, ls_libs, print_dir_diskspace, 
                             python_executable, run_realtime_process, 
                             set_process_title, shell, startfile, truepath, 
                             unixpath, vd, view_directory,) 
    from .util_class import (ReloadingMetaclass, classmember, 
                             get_comparison_methods, inject_func_as_method, 
                             inject_instance, makeForwardingMetaclass, 
                             reload_class_methods,) 
    from .util_csv import (is_float, is_int, is_list, is_str, make_csv_table, 
                           numpy_to_csv,) 
    from .util_config import (get_default_global_config, 
                              get_default_repo_config, read_repo_config, 
                              write_default_repo_config,) 
    from .util_dbg import (EmbedOnException, IPYTHON_EMBED_STR, RAISE_ALL, 
                           all_rrr, debug_exception, debug_hstack, debug_list, 
                           debug_npstack, debug_vstack, dict_dbgstr, embed, 
                           execstr_attr_list, execstr_dict, execstr_embed, 
                           execstr_func, execstr_parent_locals, execstr_src, 
                           explore_module, explore_stack, fmtlocals, formatex, 
                           get_caller_locals, get_caller_name, 
                           get_caller_prefix, get_localvar_from_stack, 
                           get_parent_frame, get_parent_globals, 
                           get_parent_locals, get_reprs, get_stack_frame, 
                           get_type, get_var_from_stack, 
                           get_varname_from_locals, get_varstr, 
                           get_varval_from_locals, haveIPython, horiz_string, 
                           import_testdata, inIPython, indent, ipython_execstr, 
                           ipython_execstr2, is_listlike, is_valid_varname, 
                           keys_dbgstr, len_dbgstr, list_allsame, list_dbgstr, 
                           load_testdata, module_functions, my_numpy_printops, 
                           pack_into, parse_locals_keylist, print_frame, 
                           print_keys, print_varlen, printex, printvar, 
                           printvar2, public_attributes, qflag, quasiquote, 
                           quit, quitflag, save_testdata, 
                           search_stack_for_localvar, search_stack_for_var, 
                           split, super_print, truncate_str, varname_regex,) 
    from .util_dev import (DEPRICATED, MemoryTracker, auto_docstr, 
                           compile_cython, disable_garbage_collection, 
                           enable_garbage_collection, find_exe, 
                           garbage_collect, get_cython_exe, get_object_base, 
                           get_object_size, get_object_size_str, get_stats, 
                           get_stats_str, info, init_catch_ctrl_c, 
                           is_developer, listinfo, make_call_graph, 
                           make_default_docstr, make_object_graph, memory_dump, 
                           myprint, npArrInfo, npinfo, numpy_list_num_bits, 
                           print_auto_docstr, print_object_size, 
                           print_object_size_tree, print_stats, printableType, 
                           printableVal, report_memsize, reset_catch_ctrl_c, 
                           testit, timeit_compare, tuples_to_unique_scalars,) 
    from .util_decor import (NOINDENT_DECOR, PROFILING, SIG_PRESERVE, 
                             UNIQUE_NUMPY, accepts_numpy, accepts_scalar_input, 
                             accepts_scalar_input2, 
                             accepts_scalar_input_vector_output, getter_1to1, 
                             getter_1toM, ignores_exc_tb, indent_func, 
                             interested, lazyfunc, memorize, 
                             on_exception_report_input, preserve_sig, 
                             show_return_value, time_func, update_wrapper, 
                             wraps,) 
    from .util_distances import (L1, L2, L2_sqrd, compute_distances, emd, 
                                 hist_isect, nearest_point,) 
    from .util_dict import (all_dict_combinations, all_dict_combinations_lbls, 
                            build_conflict_dict, dict_union, dict_union2, 
                            dict_update_newkeys, is_dicteq, 
                            items_sorted_by_value, keys_sorted_by_value, 
                            updateif_haskey,) 
    from .util_func import (general_get, general_set, uinput_1to1,) 
    from .util_grabdata import (BadZipfile, download_url, fix_dropbox_link, 
                                grab_file_url, grab_zipped_url, 
                                split_archive_ext, unarchive_file, untar_file, 
                                unzip_file,) 
    from .util_git import (PROJECT_REPO_DIRS, PROJECT_REPO_URLS, 
                           checkout_repos, ensure_repos, get_repo_dirs, 
                           get_repo_dname, gg_command, gitcmd, is_gitrepo, 
                           isdir, pull_repos, repo_list, set_project_repos, 
                           set_userid, setup_develop_repos, std_build_command,) 
    from .util_hash import (ALPHABET, ALPHABET_16, BIGBASE, HASH_LEN, 
                            augment_uuid, convert_hexstr_to_bigbase, 
                            deterministic_uuid, get_file_hash, get_file_uuid, 
                            get_zero_uuid, hashable_to_uuid, hashstr, 
                            hashstr_arr, hashstr_md5, hashstr_sha1, image_uuid, 
                            random_nonce, random_uuid,) 
    from .util_inject import (ARGV_DEBUG_FLAGS, DUMMYPROF_FUNC, KERNPROF_FUNC, 
                              PROF_FUNC_PAT_LIST, PROF_MOD_PAT_LIST, SILENT, 
                              TIMERPROF_FUNC, argv, get_injected_modules, 
                              inject, inject_all, inject_colored_exceptions, 
                              inject_print_functions, inject_profile_function, 
                              inject_reload_function, memprof,) 
    from .util_io import (load_cPkl, read_from, save_cPkl, try_decode, 
                          write_to,) 
    from .util_iter import (cycle, ensure_iterable, ichunks, ifilter_Nones, 
                            ifilter_items, ifilterfalse_items, iflatten, 
                            iflatten_scalars, interleave, interleave2, 
                            interleave3, isiterable, islice, roundrobin,) 
    from .util_inspect import (list_class_funcnames,) 
    from .util_logging import (PRINT_ALL_CALLERS, add_logging_handler, 
                               get_log_fpath, get_logging_dir, logdir_cacheid, 
                               start_logging, stop_logging,) 
    from .util_list import (alloc_lists, alloc_nones, assert_all_not_None, 
                            assert_unflat_level, debug_consec_list, 
                            depth_profile, ensure_list_size, filter_Nones, 
                            filter_items, filterfalse_items, flag_unique_items, 
                            flatten, flattenize, get_callable_name, 
                            get_dirty_items, get_list_column, 
                            intersect_ordered, invertable_flatten, 
                            invertable_flatten2, issorted, list_deep_types, 
                            list_depth, list_getat, list_replace, listfind, 
                            partial_imap_1to1, safe_listget, safe_slice, 
                            sample_zip, scalar_input_map, sortedby, tuplize, 
                            unflatten, unique_keep_order2, unique_ordered, 
                            unique_unordered,) 
    from .util_num import (commas, fewest_digits_float_str, float_to_decimal, 
                           format_, int_comma_str, num2_sigfig, num_fmt, 
                           order_of_magnitude_ceil, sigfig_str,) 
    from .util_numpy import (deterministic_sample, deterministic_shuffle, 
                             inbounds, index_of, intersect2d, 
                             intersect2d_numpy, list_index, listlike_copy, 
                             make_incrementer, npfind, random_indexes, 
                             random_sample, sample_domain, spaced_indexes, 
                             spaced_items, tiled_range,) 
    from .util_path import (IMG_EXTENSIONS, append_suffixlist_to_namelist, 
                            assert_exists, assertpath, basename_noext, 
                            checkpath, copy, copy_all, copy_list, copy_task, 
                            delete, dirsplit, ensuredir, ensurepath, 
                            existing_subpath, ext, extend_regex, file_bytes, 
                            file_megabytes, fixwin32_shortname, 
                            fnames_to_fpaths, fpaths_to_fnames, 
                            get_basename_noext_list, get_basepath_list, 
                            get_ext_list, get_module_dir, glob, grep, iglob, 
                            is_module_dir, is_private_module, isfile, islink, 
                            ismount, list_images, longest_existing_path, ls, 
                            ls_dirs, ls_moduledirs, ls_modulefiles, 
                            matches_image, move_list, newcd, num_images_in_dir, 
                            path_ndir_split, platform_path, progress_func, 
                            remove_dirs, remove_file, remove_file_list, 
                            remove_files_in_dir, symlink, tail, truepath, 
                            truepath_relative, unixjoin, win_shortcut,) 
    from .util_print import (Indenter, NO_INDENT, NpPrintOpts, filesize_str, 
                             full_numpy_repr, horiz_print, printNOTQUIET, 
                             printVERBOSE, printWARN, print_filesize, 
                             print_locals, printif, printshape,) 
    from .util_progress import (AGGROFLUSH, PROGGRESS_BACKSPACE, ProgressIter, 
                                VALID_PROGRESS_TYPES, log_progress, prog_func, 
                                progiter, progress_func, progress_str, 
                                simple_progres_func,) 
    from .util_parallel import (BACKEND, close_pool, ensure_pool, generate, 
                                get_default_numprocs, in_main_process, 
                                init_pool, init_worker, new_pool, process, 
                                set_num_procs, tic, toc,) 
    from .util_resources import (available_memory, byte_str2, 
                                 current_memory_usage, get_resource_limits, 
                                 memstats, num_cpus, peak_memory, 
                                 print_resource_usage, time_in_systemmode, 
                                 time_in_usermode, time_str2, total_memory, 
                                 used_memory,) 
    from .util_str import (GLOBAL_TYPE_ALIASES, TAU, align, align_lines, 
                           bbox_str, byte_str, byte_str2, codeblock, 
                           dict_aliased_repr, dict_itemstr_list, dict_str, 
                           extend_global_aliases, file_megabytes_str, func_str, 
                           get_freespace_str, get_indentation, 
                           get_unix_timedelta, get_unix_timedelta_str, 
                           horiz_string, hz_str, indent_list, indentjoin, 
                           joins, list_aliased_repr, list_str, listinfo_str, 
                           long_fname_format, newlined_list, 
                           order_of_magnitude_str, packstr, padded_str_range, 
                           remove_chars, seconds_str, str2, str_between, 
                           theta_str, tupstr, unindent, var_aliased_repr, 
                           verts_str,) 
    from .util_sysreq import (DEBUG, ensure_in_pythonpath, locate_path,) 
    from .util_setup import (NOOP, SETUP_PATTERNS, assert_in_setup_repo, 
                             autogen_sphinx_apidoc, build_pyo, clean, 
                             find_ext_modules, find_packages, get_cmdclass, 
                             get_numpy_include_dir, parse_author, 
                             parse_package_for_version, parse_readme, presetup, 
                             presetup_commands, read_license, setup_chmod, 
                             setuptools_setup, translate_cyth,) 
    from .util_regex import (REGEX_VARNAME, RE_FLAGS, RE_KWARGS, 
                             get_match_text, named_field, named_field_regex, 
                             regex_get_match, regex_parse, regex_replace, 
                             regex_replace_lines, regex_search, regex_split, 
                             repl_field, sed, sedfile,) 
    from .util_time import (Timer, exiftime_to_unixtime, get_day, get_month, 
                            get_timestamp, get_timestats_str, get_year, tic, 
                            toc, unixtime_to_datetime, unixtime_to_timedelta,) 
    from .util_type import (BooleanType, FloatType, IntType, LongType, 
                            NP_NDARRAY, VALID_BOOL_TYPES, VALID_FLOAT_TYPES, 
                            VALID_INT_TYPES, assert_int, bool_from_str, 
                            is_bool, is_dict, is_func_or_method, 
                            is_func_or_method_or_partial, is_funclike, 
                            is_tuple, is_type, is_valid_floattype, smart_cast, 
                            type_str,) 
    from .util_tests import (HAPPY_FACE, SAD_FACE, autogen_ibeis_runtest, 
                             autogen_run_tests, bubbletext, def_test, 
                             printTEST, run_test, tryimport,) 
    from .DynamicStruct import (AbstractPrintable, DynStruct,) 
    from .Preferences import (Pref, PrefChoice, PrefInternal, PrefNode, 
                              PrefTree,) 
    print, print_, printDBG, rrr, profile = util_inject.inject(
        __name__, '[utool]')
    
    def reload_subs():
        """ Reloads utool and submodules """
        rrr()
        getattr(_internal, 'rrr', lambda: None)()
        getattr(util_alg, 'rrr', lambda: None)()
        getattr(util_aliases, 'rrr', lambda: None)()
        getattr(util_arg, 'rrr', lambda: None)()
        getattr(util_cache, 'rrr', lambda: None)()
        getattr(util_cplat, 'rrr', lambda: None)()
        getattr(util_class, 'rrr', lambda: None)()
        getattr(util_csv, 'rrr', lambda: None)()
        getattr(util_config, 'rrr', lambda: None)()
        getattr(util_dbg, 'rrr', lambda: None)()
        getattr(util_dev, 'rrr', lambda: None)()
        getattr(util_decor, 'rrr', lambda: None)()
        getattr(util_distances, 'rrr', lambda: None)()
        getattr(util_dict, 'rrr', lambda: None)()
        getattr(util_func, 'rrr', lambda: None)()
        getattr(util_grabdata, 'rrr', lambda: None)()
        getattr(util_git, 'rrr', lambda: None)()
        getattr(util_hash, 'rrr', lambda: None)()
        getattr(util_inject, 'rrr', lambda: None)()
        getattr(util_io, 'rrr', lambda: None)()
        getattr(util_iter, 'rrr', lambda: None)()
        getattr(util_inspect, 'rrr', lambda: None)()
        getattr(util_logging, 'rrr', lambda: None)()
        getattr(util_list, 'rrr', lambda: None)()
        getattr(util_num, 'rrr', lambda: None)()
        getattr(util_numpy, 'rrr', lambda: None)()
        getattr(util_path, 'rrr', lambda: None)()
        getattr(util_print, 'rrr', lambda: None)()
        getattr(util_progress, 'rrr', lambda: None)()
        getattr(util_parallel, 'rrr', lambda: None)()
        getattr(util_resources, 'rrr', lambda: None)()
        getattr(util_str, 'rrr', lambda: None)()
        getattr(util_sysreq, 'rrr', lambda: None)()
        getattr(util_setup, 'rrr', lambda: None)()
        getattr(util_regex, 'rrr', lambda: None)()
        getattr(util_time, 'rrr', lambda: None)()
        getattr(util_type, 'rrr', lambda: None)()
        getattr(util_tests, 'rrr', lambda: None)()
        getattr(DynamicStruct, 'rrr', lambda: None)()
        getattr(Preferences, 'rrr', lambda: None)()
        rrr()
        try:
            # For utool
            dev_reimport()
        except Exception:
            pass
    rrrr = reload_subs
    # </AUTOGEN_INIT>