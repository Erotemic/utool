# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function  # , unicode_literals
from six.moves import zip, filter, filterfalse, map, range  # NOQA
import six  # NOQA
from os.path import split, dirname
from utool import util_class  # NOQA
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__, '[util_project]')


__GLOBAL_PROFILE__ = None


#@util_class.ReloadingMetaclass
class UserProfile(object):
    def __init__(self):
        self.project_dpaths = None
        self.project_include_patterns = None
        self.project_exclude_dirs = None


def ibeis_user_profile():
    import utool as ut
    import sys
    addpath = True
    module_fpath = ut.truepath('~/local/init/__REPOS1__.py')
    if addpath:
        module_dpath = dirname(module_fpath)
        sys.path.append(module_dpath)
    __REPOS1__ = ut.import_module_from_fpath(module_fpath)
    self = UserProfile()
    self.project_dpaths = __REPOS1__.PROJECT_REPOS
    self.project_dpaths += [ut.truepath('~/latex/crall-candidacy-2015/')]
    self.project_include_patterns = [
        '*.py', '*.cxx', '*.cpp', '*.hxx', '*.hpp', '*.c', '*.h', '*.vim'
    ]
    self.project_exclude_dirs = [
        '_graveyard', '_broken', 'CompilerIdCXX', 'CompilerIdC', 'build',
        'old',
    ]
    return self


def ensure_user_profile(user_profile=None):
    global __GLOBAL_PROFILE__
    if __GLOBAL_PROFILE__ is None:
        import utool as ut
        if ut.is_developer():
            __GLOBAL_PROFILE__ = ibeis_user_profile()
    if user_profile is None:
        user_profile = __GLOBAL_PROFILE__
    return user_profile


def grep_projects(tofind_list, user_profile=None, verbose=True, **kwargs):
    r"""
    Greps the projects defined in the current UserProfile

    Args:
        tofind_list (list):
        user_profile (None): (default = None)

    Kwargs:
        user_profile

    CommandLine:
        python -m utool --tf grep_projects grep_projects

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_project import *  # NOQA
        >>> import utool as ut
        >>> import sys
        >>> tofind_list = ut.get_argval('--find', type_=list, default=[sys.argv[-1]])
        >>> grep_projects(tofind_list)
    """
    import utool as ut
    user_profile = ensure_user_profile(user_profile)
    grepkw = {}
    grepkw['exclude_dirs'] = user_profile.project_exclude_dirs
    grepkw['dpath_list'] = user_profile.project_dpaths
    grepkw['include_patterns'] = user_profile.project_include_patterns
    grepkw.update(kwargs)

    msg_list1 = []
    msg_list2 = []

    print_ = msg_list1.append
    print_('Greping Projects')
    print_('tofind_list = %s' % (ut.list_str(tofind_list, nl=True),))
    print_('grepkw = %s' % ut.dict_str(grepkw, nl=True))
    if verbose:
        print('\n'.join(msg_list1))
    with ut.Timer('greping', verbose=True):
        found_fpath_list, found_lines_list, found_lxs_list = ut.grep(tofind_list, **grepkw)

    print_ = msg_list2.append
    for fpath, lines, lxs in zip(found_fpath_list, found_lines_list, found_lxs_list):
        print_('----------------------')
        print_('found %d line(s) in %r: ' % (len(lines), fpath))
        name = split(fpath)[1]
        max_line = len(lines)
        ndigits = str(len(str(max_line)))
        for (lx, line) in zip(lxs, lines):
            line = line.replace('\n', '')
            print_(('%s : %' + ndigits + 'd |%s') % (name, lx, line))

    print_('====================')
    print_('found_fpath_list = ' + ut.list_str(found_fpath_list))
    print_('')
    #print_('gvim -o ' + ' '.join(found_fpath_list))
    if verbose:
        print('\n'.join(msg_list2))
    msg_list = msg_list1 + msg_list2
    return msg_list


## Grep my projects
#def gp(r, regexp):
#    rob_nav._grep(r, [regexp], recursive=True, dpath_list=project_dpaths(), regex=True)

## Sed my projects
#def sp(r, regexpr, repl, force=False):
#    rob_nav._sed(r, regexpr, repl, force=force, recursive=True, dpath_list=project_dpaths())

def sed_projects(regexpr, repl, force=False, recursive=True, user_profile=None, **kwargs):
    """

    Args:
        regexpr (?):
        repl (?):
        force (bool): (default = False)
        recursive (bool): (default = True)
        user_profile (None): (default = None)

    CommandLine:
        python -m utool.util_project --exec-sed_projects

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_project import *  # NOQA
        >>> regexpr = ut.get_argval('--find', type_=str, default=sys.argv[-1])
        >>> repl = ut.get_argval('--repl', type_=str, default=sys.argv[-2])
        >>> force = False
        >>> recursive = True
        >>> user_profile = None
        >>> result = sed_projects(regexpr, repl, force, recursive, user_profile)
        >>> print(result)

    Ignore:
        regexpr = 'annotation match_scores'
        repl = 'draw_score_sep'

    """
    # FIXME: finishme
    import utool as ut
    user_profile = ensure_user_profile(user_profile)

    sedkw = {}
    sedkw['exclude_dirs'] = user_profile.project_exclude_dirs
    sedkw['dpath_list'] = user_profile.project_dpaths
    sedkw['include_patterns'] = user_profile.project_include_patterns
    sedkw.update(kwargs)

    msg_list1 = []
    #msg_list2 = []

    print_ = msg_list1.append
    print_('Seding Projects')
    print(' * regular expression : %r' % (regexpr,))
    print(' * replacement        : %r' % (repl,))
    print_('sedkw = %s' % ut.dict_str(sedkw, nl=True))

    print(' * recursive: %r' % (recursive,))
    print(' * force: %r' % (force,))

    # Walk through each directory recursively
    for fpath in ut.matching_fnames(sedkw['dpath_list'],
                                    sedkw['include_patterns'],
                                    sedkw['exclude_dirs'],
                                    recursive=recursive):
        ut.sedfile(fpath, regexpr, repl, force)


#def extend_regex(regexpr):
#    regex_map = {
#        r'\<': r'\b(?=\w)',
#        r'\>': r'\b(?!\w)',
#        ('UNSAFE', r'\x08'): r'\b',
#    }
#    for key, repl in six.iteritems(regex_map):
#        if isinstance(key, tuple):
#            search = key[1]
#        else:
#            search = key
#        if regexpr.find(search) != -1:
#            if isinstance(key, tuple):
#                print('WARNING! Unsafe regex with: %r' % (key,))
#            regexpr = regexpr.replace(search, repl)
#    return regexpr
#regexpr = extend_regex(regexpr)
#if '\x08' in regexpr:
#    print('Remember \\x08 != \\b')
#    print('subsituting for you for you')
#    regexpr = regexpr.replace('\x08', '\\b')
#    print(' * regular expression : %r' % (regexpr,))


if False:
    def ensure_vim_plugins():
        """

        python ~/local/init/ensure_vim_plugins.py
        '~/local/init/ensure_vim_plugins.py'
        '~/local/init/__REPOS1__.py'

        """
        # TODO
        pass

    def find_module_callers():
        """
        TODO:
        attempt to build a call graph between module functions to make it easy to see
        what can be removed and what cannot.
        """
        import utool as ut
        from os.path import normpath
        mod_fpath = ut.truepath('~/code/ibeis/ibeis/expt/results_analyzer.py')
        mod_fpath = ut.truepath('~/code/ibeis/ibeis/expt/results_all.py')
        mod_fpath = ut.truepath('~/code/ibeis/ibeis/expt/results_organizer.py')
        module = ut.import_module_from_fpath(mod_fpath)
        user_profile = ut.ensure_user_profile()
        doctestables = list(ut.iter_module_doctestable(module, include_builtin=False))
        grepkw = {}
        grepkw['exclude_dirs'] = user_profile.project_exclude_dirs
        grepkw['dpath_list'] = user_profile.project_dpaths
        grepkw['verbose'] = True

        usage_map = {}
        for funcname, func in doctestables:
            print('Searching for funcname = %r' % (funcname,))
            found_fpath_list, found_lines_list, found_lxs_list = ut.grep([funcname], **grepkw)
            used_in = (found_fpath_list, found_lines_list, found_lxs_list)
            usage_map[funcname] = used_in

        external_usage_map = {}
        for funcname, used_in in usage_map.items():
            (found_fpath_list, found_lines_list, found_lxs_list) = used_in
            isexternal_flag = [normpath(fpath) != normpath(mod_fpath) for fpath in found_fpath_list]
            ext_used_in = (ut.compress(found_fpath_list, isexternal_flag),
                           ut.compress(found_lines_list, isexternal_flag),
                           ut.compress(found_lxs_list, isexternal_flag))
            external_usage_map[funcname] = ext_used_in

        for funcname, used_in in external_usage_map.items():
            (found_fpath_list, found_lines_list, found_lxs_list) = used_in

        print('Calling modules: \n' +
              ut.repr2(ut.unique_keep_order(ut.flatten([used_in[0] for used_in in  external_usage_map.values()])), nl=True))


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_project
        python -m utool.util_project --allexamples
        python -m utool.util_project --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
