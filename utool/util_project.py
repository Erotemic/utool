# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function  # , unicode_literals
from six.moves import zip, filter, filterfalse, map, range  # NOQA
import six  # NOQA
from os.path import split, dirname, join
from utool import util_class  # NOQA
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__, '[util_project]')


__GLOBAL_PROFILE__ = None


def ensure_text(fname, text, repo_dpath='.', force=False, locals_={}):
    """
    Args:
        fname (str):  file name
        text (str):
        repo_dpath (str):  directory path string(default = '.')
        force (bool): (default = False)
        locals_ (dict): (default = {})

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_git import *  # NOQA
        >>> import utool as ut
        >>> result = setup_repo()
        >>> print(result)
    """
    print('Ensuring fname=%r' % (fname))
    import utool as ut
    fpath = join(repo_dpath, fname)
    if force or not ut.checkpath(fpath, verbose=2, n=5):
        text_ = ut.remove_codeblock_syntax_sentinals(text)
        text_ = text_.format(**locals_) + '\n'
        ut.writeto(fpath, text_)


def setup_repo():
    r"""
    Creates default structure for a new repo

    CommandLine:
        python -m utool.util_git --exec-setup_repo --repo=dtool --codedir=~/code

        python -m utool --tf setup_repo --repo=dtool --codedir=~/code
        python -m utool --tf setup_repo --repo=ibeis-flukematch-module --codedir=~/code --modname=ibeis_flukematch

        python -m utool --tf setup_repo

    Python:
        ipython
        import utool as ut
        ut.rrrr(0); ut.setup_repo()

    Example:
        >>> # SCRIPT
        >>> from utool.util_git import *  # NOQA
        >>> import utool as ut
        >>> result = setup_repo()
        >>> print(result)
    """
    print('\n [setup_repo]!')
    # import os
    from functools import partial
    import utool as ut
    # import os
    code_dpath  = ut.truepath(ut.get_argval('--code-dir', default='~/code'))
    repo_fname = (ut.get_argval(('--repo', '--repo-name'), type_=str))
    repo_dpath = join(code_dpath, repo_fname)
    modname = ut.get_argval('--modname', default=repo_fname)
    ut.ensuredir(repo_dpath, verbose=True)
    with ut.ChdirContext(repo_dpath):
        # os.chdir(repo_fname)
        locals_ = locals()
        force = True

        _ensure_text = partial(ensure_text, repo_dpath='.', force=False, locals_=locals_)

        _ensure_text(
            fname='todo.md',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                # {modname} TODO File

                * Add TODOS!
                # ENDBLOCK
                ''')
        )

        _ensure_text(
            fname='README.md',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                # {modname} README FILE
                # ENDBLOCK
                ''')
        )

        _ensure_text(
            fname='setup.py',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                #!/usr/bin/env python2.7
                from __future__ import absolute_import, division, print_function, unicode_literals
                from setuptools import setup
                try:
                    from utool import util_setup
                except ImportError:
                    print('ERROR: setup requires utool')
                    raise

                INSTALL_REQUIRES = [
                    #'cython >= 0.21.1',
                    #'numpy >= 1.9.0',
                    #'scipy >= 0.16.0',
                ]

                CLUTTER_PATTERNS = [
                    # Patterns removed by python setup.py clean
                ]

                if __name__ == '__main__':
                    kwargs = util_setup.setuptools_setup(
                        setup_fpath=__file__,
                        name='{modname}',
                        packages=util_setup.find_packages(),
                        version=util_setup.parse_package_for_version('{modname}'),
                        license=util_setup.read_license('LICENSE'),
                        long_description=util_setup.parse_readme('README.md'),
                        ext_modules=util_setup.find_ext_modules(),
                        cmdclass=util_setup.get_cmdclass(),
                        #description='description of module',
                        #url='https://github.com/<username>/{repo_fname}.git',
                        #author='<author>',
                        #author_email='<author_email>',
                        keywords='',
                        install_requires=INSTALL_REQUIRES,
                        clutter_patterns=CLUTTER_PATTERNS,
                        #package_data={{'build': ut.get_dynamic_lib_globstrs()}},
                        #build_command=lambda: ut.std_build_command(dirname(__file__)),
                        classifiers=[],
                    )
                    setup(**kwargs)
                # ENDBLOCK
                '''
            )
        )

        _ensure_text(
            fname='.gitignore',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                *.py[cod]

                # C extensions
                *.so
                # Packages
                *.egg
                *.egg-info
                dist
                build
                eggs
                parts
                bin
                var
                sdist
                develop-eggs
                .installed.cfg
                lib
                lib64
                __pycache__

                # Installer logs
                pip-log.txt

                # Print Logs
                logs

                # Unit test / coverage reports
                .coverage
                .tox
                nosetests.xml

                # Translations
                *.mo

                # Mr Developer
                .mr.developer.cfg
                .project
                .pydevproject
                .DS_Store
                *.dump.txt
                *.sqlite3

                # profiler
                *.lprof
                *.prof

                *.flann
                *.npz

                # utool output
                _test_times.txt
                failed.txt

                *.orig
                _doc
                test_times.txt
                failed_doctests.txt
                # ENDBLOCK
                '''
            )
        )

        _ensure_text(
            fname=join(repo_dpath, modname, '__init__.py'),
            text=ut.codeblock(
                r'''
                __version__ = '0.0.0'
                '''
            )
        )

    ut.ensuredir(join(repo_dpath, modname), verbose=True)


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


@profile
def grep_projects(tofind_list, user_profile=None, verbose=True, new=False, **kwargs):
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

    if new:
        return found_fpath_list, found_lines_list
    else:
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
