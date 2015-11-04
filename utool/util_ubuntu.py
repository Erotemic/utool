# -*- coding: utf-8 -*-


def add_new_mimetype_association(ext, mime_name, exe_fpath=None, dry=True):
    """
    TODO: move to external manager and generalize

    Args:
        ext (str): extension to associate
        mime_name (str): the name of the mime_name to create (defaults to ext)
        exe_fpath (str): executable location if this is for one specific file

    References:
        https://wiki.archlinux.org/index.php/Default_applications#Custom_file_associations

    Args:
        ext (str): extension to associate
        exe_fpath (str): executable location
        mime_name (str): the name of the mime_name to create (defaults to ext)

    CommandLine:
        python -m utool.util_ubuntu --exec-add_new_mimetype_association
        python -m utool.util_ubuntu --exec-add_new_mimetype_association --mime-name=ipynb+json --ext=.ipynb --exe-fpath=/usr/local/bin/ipynb

    Example:
        >>> # SCRIPT
        >>> from utool.util_ubuntu import *  # NOQA
        >>> ext = ut.get_argval('--ext', type_=str, default=None)
        >>> mime_name = ut.get_argval('--mime_name', type_=str, default=None)
        >>> exe_fpath = ut.get_argval('--exe_fpath', type_=str, default=None)
        >>> dry = not ut.get_argflag('--force')
        >>> result = add_new_mimetype_association(ext, mime_name, exe_fpath, dry)
        >>> print(result)
    """
    import utool as ut
    from os.path import join, splitext, basename
    terminal = True

    mime_codeblock = ut.codeblock(
        '''
        <?xml version="1.0" encoding="UTF-8"?>
        <mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
            <mime-type type="application/x-{mime_name}">
                <glob-deleteall/>
                <glob pattern="*{ext}"/>
            </mime-type>
        </mime-info>
        '''
    ).format(**locals())

    prefix = ut.truepath('~/.local/share')
    mime_dpath = join(prefix, 'mime/packages')
    mime_fpath = join(mime_dpath, 'application-x-{mime_name}.xml'.format(**locals()))

    print(mime_codeblock)
    print('---')
    print(mime_fpath)
    print('L___')

    if exe_fpath is not None:
        exe_fname_noext = splitext(basename(exe_fpath))[0]
        app_name = exe_fname_noext.replace('_', '-')
        nice_name = ' '.join(
            [word[0].upper() + word[1:].lower()
             for word in app_name.replace('-', ' ').split(' ')]
        )
        app_codeblock = ut.codeblock(
            '''
            [Desktop Entry]
            Name={nice_name}
            Exec={exe_fpath}
            MimeType=application/x-{mime_name}
            Terminal={terminal}
            Type=Application
            Categories=Utility;Application;
            Comment=Custom App
            '''
        ).format(**locals())
        app_dpath = join(prefix, 'applications')
        app_fpath = join(app_dpath, '{app_name}.desktop'.format(**locals()))

        print(app_codeblock)
        print('---')
        print(app_fpath)
        print('L___')

    # WRITE FILES
    if not dry:
        ut.ensuredir(mime_dpath)
        ut.ensuredir(app_dpath)

        ut.writeto(mime_fpath, mime_codeblock, verbose=ut.NOT_QUIET, n=None)
        if exe_fpath is not None:
            ut.writeto(app_fpath, app_codeblock, verbose=ut.NOT_QUIET, n=None)

        # UPDATE BACKENDS

        #ut.cmd('update-mime-database /usr/share/mime')
        #~/.local/share/applications/mimeapps.list
        print(ut.codeblock(
            '''
            Run these commands:
            update-desktop-database ~/.local/share/applications
            update-mime-database ~/.local/share/mime
            '''
        ))
        if exe_fpath is not None:
            ut.cmd('update-desktop-database ~/.local/share/applications')
        ut.cmd('update-mime-database ~/.local/share/mime')
    else:
        print('dry_run')


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_ubuntu
        python -m utool.util_ubuntu --allexamples
        python -m utool.util_ubuntu --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
