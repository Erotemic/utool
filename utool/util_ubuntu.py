# -*- coding: utf-8 -*-


def add_new_mimetype_association(ext, exe_fpath, mime_name):
    """
    TODO: move to external manager and generalize

    Args:
        ext (str): extension to associate
        exe_fpath (str): executable location
        mime_name (str): the name of the mime_name to create (defaults to ext)

    References:
        https://wiki.archlinux.org/index.php/Default_applications#Custom_file_associations
    """
    import utool as ut
    from os.path import join, splitext, basename
    terminal = True

    exe_fname_noext = splitext(basename(exe_fpath))[0]
    app_name = exe_fname_noext.replace('_', '-')
    nice_name = ' '.join(
        [word[0].upper() + word[1:].lower()
         for word in app_name.replace('-', ' ').split(' ')]
    )

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

    # CONFIGURE PATHS
    #prefix = '/usr/share'  # for global install
    prefix = ut.truepath('~/.local/share')
    mime_dpath = join(prefix, 'mime/packages')
    app_dpath = join(prefix, 'applications')

    # CONFIGURE NAMES

    mime_fpath = join(mime_dpath, 'application-x-{mime_name}.xml'.format(**locals()))
    app_fpath = join(app_dpath, '{app_name}.desktop'.format(**locals()))

    print(mime_codeblock)
    print('---')
    print(mime_fpath)
    print('L___')

    print(app_codeblock)
    print('---')
    print(app_fpath)
    print('L___')

    # WRITE FIELS

    ut.ensuredir(mime_dpath)
    ut.ensuredir(app_dpath)

    ut.writeto(mime_fpath, mime_codeblock, verbose=ut.NOT_QUIET, n=None)
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
    ut.cmd('update-desktop-database ~/.local/share/applications')
    ut.cmd('update-mime-database ~/.local/share/mime')
