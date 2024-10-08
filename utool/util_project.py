# -*- coding: utf-8 -*-
"""
Ignore:
    ~/local/init/REPOS1.py
"""
from __future__ import absolute_import, division, print_function  # , unicode_literals
from os.path import dirname, join
from utool import util_dev
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


__GLOBAL_PROFILE__ = None


def ensure_text(fname, text, repo_dpath='.', force=None, locals_={}, chmod=None):
    """
    Args:
        fname (str):  file name
        text (str):
        repo_dpath (str):  directory path string(default = '.')
        force (bool): (default = False)
        locals_ (dict): (default = {})

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_project import *  # NOQA
        >>> import utool as ut
        >>> result = setup_repo()
        >>> print(result)
    """
    import utool as ut
    ut.colorprint('Ensuring fname=%r' % (fname), 'yellow')

    # if not fname.endswith('__init__.py'):
    #     # HACK
    #     return

    if force is None and ut.get_argflag('--force-%s' % (fname,)):
        force = True
    text_ = ut.remove_codeblock_syntax_sentinals(text)
    fmtkw = locals_.copy()
    fmtkw['fname'] = fname
    text_ = text_.format(**fmtkw) + '\n'

    fpath = join(repo_dpath, fname)
    ut.dump_autogen_code(fpath, text_)

    # if force or not ut.checkpath(fpath, verbose=2, n=5):
    #     ut.writeto(fpath, text_)
    #     try:
    #         if chmod:
    #             ut.chmod(fpath, chmod)
    #     except Exception as ex:
    #         ut.printex(ex, iswarning=True)
    # else:
    #     print(ut.color_diff_text(ut.difftext(ut.readfrom(fpath), text_)))
    #     print('use -w to force write')


class SetupRepo(object):
    """
    Maybe make a new interface to SetupRepo?

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_project import *  # NOQA
        >>> import utool as ut
        >>> self = SetupRepo()
        >>> print(self)
    """

    def __init__(self):
        import utool as ut
        self.modname = None
        code_dpath = ut.truepath(ut.get_argval('--code-dir', default='~/code'))
        self.code_dpath = ut.unexpanduser(code_dpath)
        self.repo_fname = (ut.get_argval(('--repo', '--repo-name'), type_=str))
        self.repo_dpath = join(code_dpath, self.repo_fname)
        self.modname = ut.get_argval('--modname', default=self.repo_fname)
        self.regenfmt = 'python -m utool SetupRepo.{cmd} --modname={modname} --repo={repo_fname} --codedir={code_dpath}'
        ut.ensuredir(self.repo_dpath, verbose=True)

    def all(self):
        pass

    def ensure_text(self, fname, text, **kwargs):
        ensure_text(fname, text, locals_=self.__dict__,
                    repo_dpath=self.repo_dpath, **kwargs)

    def main(self):
        """
        python -m utool SetupRepo.main --modname=sklearn --repo=scikit-learn --codedir=~/code -w
        python -m utool SetupRepo.main --repo=ubelt --codedir=~/code --modname=ubelt -w

        Example:
        >>> # DISABLE_DOCTEST
            >>> # SCRIPT
            >>> from utool.util_project import *  # NOQA
            >>> SetupRepo().main()
        """
        self.regencmd = self.regenfmt.format(cmd='main', **self.__dict__)
        import utool as ut
        self.ensure_text(
            fname=join(self.modname, '__main__.py'),
            chmod='+x',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                #!/usr/bin/env python
                # -*- coding: utf-8 -*-
                """
                Initially Generated By:
                    {regencmd}
                """
                from __future__ import absolute_import, division, print_function, unicode_literals


                def {modname}_main():
                    ignore_prefix = []
                    ignore_suffix = []
                    import utool as ut
                    ut.main_function_tester('{modname}', ignore_prefix, ignore_suffix)

                if __name__ == '__main__':
                    """
                    Usage:
                        python -m {modname} <funcname>
                    """
                    print('Running {modname} main')
                    {modname}_main()
                # ENDBLOCK
                '''
            )
        )


def setup_repo():
    r"""
    Creates default structure for a new repo

    CommandLine:
        python -m utool setup_repo --repo=dtool --codedir=~/code

        python -m utool setup_repo --repo=dtool --codedir=~/code
        python -m utool setup_repo --repo=ibeis-flukematch-module --codedir=~/code --modname=ibeis_flukematch
        python -m utool setup_repo --repo=mtgmonte --codedir=~/code --modname=mtgmonte
        python -m utool setup_repo --repo=pydarknet --codedir=~/code --modname=pydarknet
        python -m utool setup_repo --repo=sandbox_utools --codedir=~/code --modname=sandbox_utools

        python -m utool setup_repo --repo=ubelt --codedir=~/code --modname=ubelt -w

        python -m utool setup_repo

    Python:
        ipython
        import utool as ut
        ut.rrrr(0); ut.setup_repo()

    Example:
        >>> # DISABLE_DOCTEST
        >>> # SCRIPT
        >>> from utool.util_project import *  # NOQA
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
    _code_dpath = ut.unexpanduser(code_dpath)
    repo_fname = (ut.get_argval(('--repo', '--repo-name'), type_=str))
    repo_dpath = join(code_dpath, repo_fname)
    modname = ut.get_argval('--modname', default=repo_fname)
    ut.ensuredir(repo_dpath, verbose=True)
    _regencmd = 'python -m utool --tf setup_repo --repo={repo_fname} --codedir={_code_dpath} --modname={modname}'
    flake8_noqacmd = 'flake8' + ':noqa'
    regencmd = _regencmd.format(**locals())
    with ut.ChdirContext(repo_dpath):
        # os.chdir(repo_fname)
        locals_ = locals()
        force = True

        _ensure_text = partial(ensure_text, repo_dpath='.', force=None, locals_=locals_)

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
            chmod='+x',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                #!/usr/bin/env python
                """
                Initially Generated By:
                    {regencmd} --force-{fname}
                """
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
                _timeings.txt
                failed.txt

                *.orig
                _doc
                timeings.txt
                failed_doctests.txt
                # ENDBLOCK
                '''
            )
        )

        _ensure_text(
            fname=join(repo_dpath, modname, '__init__.py'),
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                # -*- coding: utf-8 -*-
                # {flake8_noqacmd}
                """
                Initially Generated By:
                    {regencmd}
                """
                from __future__ import absolute_import, division, print_function, unicode_literals
                import sys
                __version__ = '0.0.0'

                IMPORT_TUPLES = [
                    # ('<modname>', None),
                ]
                __DYNAMIC__ = '--nodyn' not in sys.argv

                """
                python -c "import {modname}" --dump-{modname}-init
                python -c "import {modname}" --update-{modname}-init
                """

                DOELSE = False
                if __DYNAMIC__:
                    # Dynamically import listed util libraries and their members.
                    from utool._internal import util_importer
                    ignore_endswith = []
                    import_execstr = util_importer.dynamic_import(
                        __name__, IMPORT_TUPLES, ignore_endswith=ignore_endswith)
                    exec(import_execstr)
                    DOELSE = False
                else:
                    DOELSE = True
                if DOELSE:
                    # <AUTOGEN_INIT>
                    pass
                    # </AUTOGEN_INIT>
                # ENDBLOCK
                '''
            )
        )

        _ensure_text(
            fname=join(repo_dpath, modname, '__main__.py'),
            chmod='+x',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                #!/usr/bin/env python
                # -*- coding: utf-8 -*-
                """
                Initially Generated By:
                    {regencmd}
                """
                from __future__ import absolute_import, division, print_function, unicode_literals


                def {modname}_main():
                    ignore_prefix = []
                    ignore_suffix = []
                    import utool as ut
                    ut.main_function_tester('{modname}', ignore_prefix, ignore_suffix)

                if __name__ == '__main__':
                    """
                    Usage:
                        python -m {modname} <funcname>
                    """
                    print('Running {modname} main')
                    {modname}_main()
                # ENDBLOCK
                '''
            )
        )

        _ensure_text(
            fname='run_tests.py',
            chmod='+x',
            text=ut.codeblock(
                r'''
                # STARTBLOCK
                #!/usr/bin/env python
                """
                Initially Generated By:
                    {regencmd} --force-{fname}
                """
                from __future__ import absolute_import, division, print_function
                import sys
                import utool as ut


                def run_tests():
                    # Build module list and run tests
                    import sys
                    ut.change_term_title('RUN {modname} TESTS')
                    exclude_doctests_fnames = set([
                    ])
                    exclude_dirs = [
                        '_broken', 'old', 'tests', 'timeits',
                        '_scripts', '_timeits', '_doc', 'notebook',
                    ]
                    dpath_list = ['{modname}']
                    doctest_modname_list = ut.find_doctestable_modnames(
                        dpath_list, exclude_doctests_fnames, exclude_dirs)

                    coverage = ut.get_argflag(('--coverage', '--cov',))
                    if coverage:
                        import coverage
                        cov = coverage.Coverage(source=doctest_modname_list)
                        cov.start()
                        print('Starting coverage')

                        exclude_lines = [
                            'pragma: no cover',
                            'def __repr__',
                            'if self.debug:',
                            'if settings.DEBUG',
                            'raise AssertionError',
                            'raise NotImplementedError',
                            'if 0:',
                            'if ut.VERBOSE',
                            'if _debug:',
                            'if __name__ == .__main__.:',
                            'print(.*)',
                        ]
                        for line in exclude_lines:
                            cov.exclude(line)

                    for modname in doctest_modname_list:
                        exec('import ' + modname, globals())
                    module_list = [sys.modules[name] for name in doctest_modname_list]

                    nPass, nTotal, failed_cmd_list = ut.doctest_module_list(module_list)

                    if coverage:
                        print('Stoping coverage')
                        cov.stop()
                        print('Saving coverage')
                        cov.save()
                        print('Generating coverage html report')
                        cov.html_report()

                    if nPass != nTotal:
                        return 1
                    else:
                        return 0

                if __name__ == '__main__':
                    import multiprocessing
                    multiprocessing.freeze_support()
                    retcode = run_tests()
                    sys.exit(retcode)
                # ENDBLOCK
                '''
            )
        )

    ut.ensuredir(join(repo_dpath, modname), verbose=True)


class UserProfile(util_dev.NiceRepr):
    def __nice__(self):
        num_repos = 0 if self.project_dpaths is None else len(self.project_dpaths)
        return str(self.project_name) + ' %d repos' % (num_repos,)

    def __init__(self, name=None):
        self.project_name = name
        self.project_dpaths = None
        self.project_include_patterns = None
        self.project_exclude_dirs = []
        self.project_exclude_patterns = []

    def grep(self, *args, **kwargs):
        return grep_projects(user_profile=self, *args, **kwargs)

    def glob(self, *args, **kwargs):
        r"""
        # Ensure that .gitignore has certain lines
        git_ignore_lines = [
            'timeings.txt'
        ]
        fpath_list = profile.glob('.gitignore', recursive=False)
        for fpath in fpath_list:
            lines = ut.readfrom(fpath, verbose=False).split('\n')
            lines = [line.strip() for line in lines]
            missing = ut.setdiff(git_ignore_lines, lines)
            if missing:
                print('fpath = %r' % (fpath,))
                ut.writeto(fpath, '\n'.join(lines + missing))
        """
        return glob_projects(user_profile=self, *args, **kwargs)

    # def __str__(self):
    #     return


def ibeis_user_profile():
    import utool as ut
    import sys
    addpath = True
    module_fpath = ut.truepath('~/local/init/REPOS1.py')
    if addpath:
        module_dpath = dirname(module_fpath)
        sys.path.append(module_dpath)
    REPOS1 = ut.import_module_from_fpath(module_fpath)
    self = UserProfile(name='ibeis')
    #self.project_dpaths = REPOS1.PROJECT_REPOS
    self.project_dpaths = REPOS1.IBEIS_REPOS
    # self.project_dpaths += [ut.truepath('~/latex/crall-candidacy-2015/')]
    self.project_dpaths += [
        ut.truepath('~/local'),
        ut.truepath('~/code/fixtex'),
        ut.truepath('~/code/pyrf'),
        ut.truepath('~/code/detecttools'),
        ut.truepath('~/code/pydarknet'),
    ]
    self.project_dpaths = ut.unique(self.project_dpaths)
    # self.project_dpaths += [ut.truepath('~/local/vim/rc')]
    self.project_include_patterns = [
        '*.py', '*.cxx', '*.cpp', '*.hxx', '*.hpp', '*.c', '*.h', '*.vim'
        #'*.py',  # '*.cxx', '*.cpp', '*.hxx', '*.hpp', '*.c', '*.h', '*.vim'
    ]
    self.project_exclude_dirs = [
        '_graveyard', '_broken', 'CompilerIdCXX', 'CompilerIdC', 'build',
        'old', '_old_qt_hs_matcher', 'htmlcov'
    ]
    self.project_exclude_patterns = ['_grave*', '_autogen_explicit_controller*']
    return self


def ensure_user_profile(user_profile=None):
    r"""
    Args:
        user_profile (UserProfile): (default = None)

    Returns:
        UserProfile: user_profile

    CommandLine:
        python -m utool.util_project --exec-ensure_user_profile --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_project import *  # NOQA
        >>> import utool as ut
        >>> user_profile = None
        >>> user_profile = ensure_user_profile(user_profile)
        >>> result = ('user_profile = %s' % (ut.repr2(user_profile),))
        >>> print(ut.repr3(user_profile.project_dpaths))
        >>> print(result)
    """
    global __GLOBAL_PROFILE__
    if __GLOBAL_PROFILE__ is None:
        import utool as ut
        if ut.is_developer():
            __GLOBAL_PROFILE__ = ibeis_user_profile()
        else:
            __GLOBAL_PROFILE__ = UserProfile('default')
    if user_profile is None:
        user_profile = __GLOBAL_PROFILE__
    return user_profile


def grep_projects(tofind_list, user_profile=None, verbose=True, new=False,
                  **kwargs):
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
        >>> tofind_list = ut.get_argval('--find', type_=list,
        >>>                             default=[sys.argv[-1]])
        >>> grep_projects(tofind_list)
    """
    import utool as ut
    user_profile = ensure_user_profile(user_profile)
    print('user_profile = {!r}'.format(user_profile))

    kwargs = kwargs.copy()
    colored = kwargs.pop('colored', True)

    grepkw = {}
    grepkw['greater_exclude_dirs'] = user_profile.project_exclude_dirs
    grepkw['exclude_dirs'] = user_profile.project_exclude_dirs
    grepkw['dpath_list'] = user_profile.project_dpaths
    grepkw['include_patterns'] = user_profile.project_include_patterns
    grepkw['exclude_patterns'] = user_profile.project_exclude_patterns
    grepkw.update(kwargs)

    msg_list1 = []
    msg_list2 = []

    print_ = msg_list1.append
    print_('Greping Projects')
    print_('tofind_list = %s' % (ut.repr4(tofind_list, nl=True),))
    #print_('grepkw = %s' % ut.repr4(grepkw, nl=True))
    if verbose:
        print('\n'.join(msg_list1))
    #with ut.Timer('greping', verbose=True):
    grep_result = ut.grep(tofind_list, **grepkw)
    found_fpath_list, found_lines_list, found_lxs_list = grep_result

    # HACK, duplicate behavior. TODO: write grep print result function
    reflags = grepkw.get('reflags', 0)
    _exprs_flags = [ut.extend_regex2(expr, reflags)
                    for expr in tofind_list]
    extended_regex_list = ut.take_column(_exprs_flags, 0)
    reflags_list = ut.take_column(_exprs_flags, 1)
    # HACK
    # pat = ut.util_regex.regex_or(extended_regex_list)
    reflags = reflags_list[0]

    # from utool import util_regex
    resultstr = ut.make_grep_resultstr(grep_result, extended_regex_list,
                                       reflags, colored=colored)
    msg_list2.append(resultstr)
    print_ = msg_list2.append
    #for fpath, lines, lxs in zip(found_fpath_list, found_lines_list,
    #                             found_lxs_list):
    #    print_('----------------------')
    #    print_('found %d line(s) in %r: ' % (len(lines), fpath))
    #    name = split(fpath)[1]
    #    max_line = len(lines)
    #    ndigits = str(len(str(max_line)))
    #    for (lx, line) in zip(lxs, lines):
    #        line = line.replace('\n', '')
    #        print_(('%s : %' + ndigits + 'd |%s') % (name, lx, line))
    # iter_ = zip(found_fpath_list, found_lines_list, found_lxs_list)
    # for fpath, lines, lxs in iter_:
    #     print_('----------------------')
    #     print_('found %d line(s) in %r: ' % (len(lines), fpath))
    #     name = split(fpath)[1]
    #     max_line = len(lines)
    #     ndigits = str(len(str(max_line)))
    #     for (lx, line) in zip(lxs, lines):
    #         line = line.replace('\n', '')
    #         colored_line = ut.highlight_regex(
    #             line.rstrip('\n'), pat, reflags=reflags)
    #         print_(('%s : %' + ndigits + 'd |%s') % (name, lx, colored_line))

    print_('====================')
    print_('found_fpath_list = ' + ut.repr4(found_fpath_list))
    print_('')
    #print_('gvim -o ' + ' '.join(found_fpath_list))
    if verbose:
        print('\n'.join(msg_list2))
    msg_list = msg_list1 + msg_list2

    if new:
        return GrepResult(found_fpath_list, found_lines_list, found_lxs_list,
                          extended_regex_list, reflags)
    else:
        return msg_list


def glob_projects(pat, user_profile=None, recursive=True):
    """

    def testenv(modname, funcname):
        ut.import_modname(modname)
        exec(ut.execstr_funckw(table.get_rowid), globals())

    Ignore:
        >>> import utool as ut
        >>> ut.testenv('utool.util_project', 'glob_projects', globals())
        >>> from utool.util_project import *  # NOQA
    """
    import utool as ut  # NOQA
    user_profile = ensure_user_profile(user_profile)
    glob_results = ut.flatten([ut.glob(dpath, pat, recursive=recursive,
                                       exclude_dirs=user_profile.project_exclude_dirs)
                               for dpath in user_profile.project_dpaths])
    return glob_results


class GrepResult(util_dev.NiceRepr):
    def __init__(self, found_fpath_list, found_lines_list,
                 found_lxs_list, extended_regex_list, reflags):
        self.found_fpath_list = found_fpath_list
        self.found_lines_list = found_lines_list
        self.found_lxs_list = found_lxs_list
        self.extended_regex_list = extended_regex_list
        self.reflags = reflags
        self.filter_pats = []

    def __nice__(self):
        return '(%d)' % (len(self),)

    def __str__(self):
        return self.make_resultstr()

    def __len__(self):
        return len(self.found_fpath_list)

    def __delitem__(self, index):
        import utool as ut
        index = ut.ensure_iterable(index)
        ut.delete_items_by_index(self.found_fpath_list, index)
        ut.delete_items_by_index(self.found_lines_list, index)
        ut.delete_items_by_index(self.found_lxs_list, index)

    def __getitem__(self, index):
        return (
            ut.take(self.found_fpath_list, index),
            ut.take(self.found_lines_list, index),
            ut.take(self.found_lxs_list, index),
        )

    def remove_results(self, indicies):
        del self[indicies]

    def make_resultstr(self, colored=True):
        import utool as ut
        tup = (self.found_fpath_list, self.found_lines_list,
               self.found_lxs_list)
        return ut.make_grep_resultstr(tup, self.extended_regex_list,
                                      self.reflags, colored=colored)

    def pattern_filterflags(self, filter_pat):
        self.filter_pats.append(filter_pat)
        import re
        flags_list = [[re.search(filter_pat, line) is None for line in lines]
                      for fpath, lines, lxs in zip(self.found_fpath_list, self.found_lines_list, self.found_lxs_list)]
        return flags_list

    def inplace_filter_results(self, filter_pat):
        import utool as ut
        self.filter_pats.append(filter_pat)
        # Get zipflags
        flags_list = self.pattern_filterflags(filter_pat)
        # Check to see if there are any survivors
        flags = ut.lmap(any, flags_list)
        #
        found_lines_list = ut.zipcompress(self.found_lines_list, flags_list)
        found_lxs_list = ut.zipcompress(self.found_lxs_list, flags_list)
        #
        found_fpath_list = ut.compress(self.found_fpath_list, flags)
        found_lines_list = ut.compress(found_lines_list, flags)
        found_lxs_list = ut.compress(found_lxs_list, flags)
        # In place modification
        self.found_fpath_list = found_fpath_list
        self.found_lines_list = found_lines_list
        self.found_lxs_list = found_lxs_list

    def hack_remove_pystuff(self):
        import utool as ut
        # Hack of a method
        new_lines = []
        for lines in self.found_lines_list:
            # remove comment results
            flags = [not line.strip().startswith('# ') for line in lines]
            lines = ut.compress(lines, flags)

            # remove doctest results
            flags = [not line.strip().startswith('>>> ') for line in lines]
            lines = ut.compress(lines, flags)

            # remove cmdline tests
            import re
            flags = [not re.search('--test-' + self.extended_regex_list[0], line) for line in lines]
            lines = ut.compress(lines, flags)

            flags = [not re.search('--exec-' + self.extended_regex_list[0], line) for line in lines]
            lines = ut.compress(lines, flags)

            flags = [not re.search('--exec-[a-zA-z]*\\.' + self.extended_regex_list[0], line) for line in lines]
            lines = ut.compress(lines, flags)

            flags = [not re.search('--test-[a-zA-z]*\\.' + self.extended_regex_list[0], line) for line in lines]
            lines = ut.compress(lines, flags)

            # remove func defs
            flags = [not re.search('def ' + self.extended_regex_list[0], line) for line in lines]
            lines = ut.compress(lines, flags)
            new_lines += [lines]
        self.found_lines_list = new_lines

        # compress self
        flags = [len(lines_) > 0 for lines_ in self.found_lines_list]
        idxs = ut.list_where(ut.not_list(flags))
        del self[idxs]


def sed_projects(regexpr, repl, force=False, recursive=True, user_profile=None, **kwargs):
    r"""

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
        repl = 'draw_annot_scoresep'

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
    print_('sedkw = %s' % ut.repr4(sedkw, nl=True))

    print(' * recursive: %r' % (recursive,))
    print(' * force: %r' % (force,))

    # Walk through each directory recursively
    for fpath in ut.matching_fpaths(sedkw['dpath_list'],
                                    sedkw['include_patterns'],
                                    sedkw['exclude_dirs'],
                                    recursive=recursive):
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
