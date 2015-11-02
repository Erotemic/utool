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

def sed_projects(r, regexpr, repl, force=False, recursive=True, user_profile=None, **kwargs):
    # FIXME: finishme
    import utool as ut
    user_profile = ensure_user_profile(user_profile)

    #_grep(r, [repl], dpath_list=dpath_list, recursive=recursive)
    include_patterns = ['*.py', '*.cxx', '*.cpp', '*.hxx', '*.hpp', '*.c', '*.h']
    dpath_list = user_profile.dpath_list
    print('sed-ing %r' % (dpath_list,))
    print(' * regular include_patterns : %r' % (include_patterns,))
    print(' * regular expression : %r' % (regexpr,))
    print(' * replacement        : %r' % (repl,))
    print(' * recursive: %r' % (recursive,))
    print(' * force: %r' % (force,))

    def extend_regex(regexpr):
        regex_map = {
            r'\<': r'\b(?=\w)',
            r'\>': r'\b(?!\w)',
            ('UNSAFE', r'\x08'): r'\b',
        }
        for key, repl in six.iteritems(regex_map):
            if isinstance(key, tuple):
                search = key[1]
            else:
                search = key
            if regexpr.find(search) != -1:
                if isinstance(key, tuple):
                    print('WARNING! Unsafe regex with: %r' % (key,))
                regexpr = regexpr.replace(search, repl)
        return regexpr
    regexpr = extend_regex(regexpr)
    if '\x08' in regexpr:
        print('Remember \\x08 != \\b')
        print('subsituting for you for you')
        regexpr = regexpr.replace('\x08', '\\b')
        print(' * regular expression : %r' % (regexpr,))

    # Walk through each directory recursively
    # FIXME
    for fpath in ut.matc_matching_fnames(dpath_list, include_patterns, recursive=recursive):
        ut.sedfile(fpath, regexpr, repl, force)


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
