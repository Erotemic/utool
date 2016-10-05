# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join, splitext, basename
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


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
        # Add ability to open ipython notebooks via double click
        python -m utool.util_ubuntu --exec-add_new_mimetype_association --mime-name=ipynb+json --ext=.ipynb --exe-fpath=/usr/local/bin/ipynb
        python -m utool.util_ubuntu --exec-add_new_mimetype_association --mime-name=ipynb+json --ext=.ipynb --exe-fpath=jupyter-notebook --force

        python -m utool.util_ubuntu --exec-add_new_mimetype_association --mime-name=sqlite --ext=.sqlite --exe-fpath=sqlitebrowser

    Example:
        >>> # SCRIPT
        >>> from utool.util_ubuntu import *  # NOQA
        >>> import utool as ut
        >>> ext = ut.get_argval('--ext', type_=str, default=None)
        >>> mime_name = ut.get_argval('--mime_name', type_=str, default=None)
        >>> exe_fpath = ut.get_argval('--exe_fpath', type_=str, default=None)
        >>> dry = not ut.get_argflag('--force')
        >>> result = add_new_mimetype_association(ext, mime_name, exe_fpath, dry)
        >>> print(result)
    """
    import utool as ut
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


def make_application_icon(exe_fpath, dry=True, props={}):
    r"""
    CommandLine:
        python -m utool.util_ubuntu --exec-make_application_icon --exe=cockatrice --icon=/home/joncrall/code/Cockatrice/cockatrice/resources/cockatrice.png
        python -m utool.util_ubuntu --exec-make_application_icon --exe=cockatrice --icon=/home/joncrall/code/Cockatrice/cockatrice/resources/cockatrice.png
        python -m utool.util_ubuntu --exec-make_application_icon --exe=/opt/zotero/zotero --icon=/opt/zotero/chrome/icons/default/main-window.ico

        python -m utool.util_ubuntu --exec-make_application_icon --exe "env WINEPREFIX="/home/joncrall/.wine" wine C:\\\\windows\\\\command\\\\start.exe /Unix /home/joncrall/.wine32-dotnet45/dosdevices/c:/users/Public/Desktop/Hearthstone.lnk" --path "/home/joncrall/.wine/dosdevices/c:/Program Files (x86)/Hearthstone"
        # Exec=env WINEPREFIX="/home/joncrall/.wine" wine /home/joncrall/.wine/drive_c/Program\ Files\ \(x86\)/Battle.net/Battle.net.exe

        --icon=/opt/zotero/chrome/icons/default/main-window.ico

        python -m utool.util_ubuntu --exec-make_application_icon --exe=/home/joncrall/code/build-ArenaTracker-Desktop_Qt_5_6_1_GCC_64bit-Debug

        update-desktop-database ~/.local/share/applications


    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_ubuntu import *  # NOQA
        >>> import utool as ut
        >>> exe_fpath = ut.get_argval('--exe', default='cockatrice')
        >>> icon = ut.get_argval('--icon', default=None)
        >>> dry = not ut.get_argflag(('--write', '-w'))
        >>> props = {'terminal': False, 'icon': icon}
        >>> result = make_application_icon(exe_fpath, dry, props)
        >>> print(result)
    """
    import utool as ut
    exe_fname_noext = splitext(basename(exe_fpath))[0]
    app_name = exe_fname_noext.replace('_', '-')
    nice_name = ' '.join(
        [word[0].upper() + word[1:].lower()
         for word in app_name.replace('-', ' ').split(' ')]
    )
    lines = [
        '[Desktop Entry]',
        'Name={nice_name}',
        'Exec={exe_fpath}',
    ]

    if 'mime_name' in props:
        lines += ['MimeType=application/x-{mime_name}']

    if 'icon' in props:
        lines += ['Icon={icon}']

    if props.get('path'):
        lines += ['Path={path}']

    # if props.get('comment'):
    #     lines += ['Path={comment}']

    lines += [
        'Terminal={terminal}',
        'Type=Application',
        'Categories=Utility;Application;',
        'Comment=Custom App',
    ]
    fmtdict = locals()
    fmtdict.update(props)

    prefix = ut.truepath('~/.local/share')
    app_codeblock = '\n'.join(lines).format(**fmtdict)
    app_dpath = join(prefix, 'applications')
    app_fpath = join(app_dpath, '{app_name}.desktop'.format(**locals()))

    print(app_codeblock)
    print('---')
    print(app_fpath)
    print('L___')

    if not dry:
        ut.writeto(app_fpath, app_codeblock, verbose=ut.NOT_QUIET, n=None)
        ut.cmd('update-desktop-database ~/.local/share/applications')


class XCtrl(object):
    """
    xdotool key ctrl+shift+i

    References:
        http://superuser.com/questions/382616/detecting-currently-active-window
        http://askubuntu.com/questions/455762/xbindkeys-wont-work-properly

    Ignore:
        xdotool keyup --window 0 7 type --clearmodifiers ---window 0 '%paste'

        # List current windows:
        wmctrl  -l

        # Get current window
        xdotool getwindowfocus getwindowname


        #====
        # Get last opened window
        #====

        win_title=x-terminal-emulator.X-terminal-emulator
        key_ = 'x-terminal-emulator.X-terminal-emulator'

        # Get all windows in current workspace
        workspace_number=`wmctrl -d | grep '\*' | cut -d' ' -f 1`
        win_list=`wmctrl -lx | grep $win_title | grep " $workspace_number " | awk '{print $1}'`

        # Get stacking order of windows in current workspace
        win_order=$(xprop -root|grep "^_NET_CLIENT_LIST_STACKING" | tr "," " ")

        echo $win_list



    CommandLine:
        python -m utool.util_ubuntu XCtrl

    Example:
        >>> # Script
        >>> import utool as ut
        >>> from utool import util_ubuntu
        >>> orig_window = []
        >>> ut.copy_text_to_clipboard(ut.lorium_ipsum())
        >>> doscript = [
        >>>     ('focus', 'x-terminal-emulator.X-terminal-emulator'),
        >>>     ('type', '%paste'),
        >>>     ('key', 'KP_Enter'),
        >>>    # ('focus', 'GVIM')
        >>> ]
        >>> util_ubuntu.XCtrl.do(*doscript, sleeptime=.01)

    Ignore:
        >>> ut.copy_text_to_clipboard(text)
        >>> if '\n' in text or len(text) > 20:
        >>>     text = '\'%paste\''
        >>> else:
        >>>     import pipes
        >>>     text = pipes.quote(text.lstrip(' '))
        >>>     ('focus', 'GVIM'),
        >>> #
        >>> doscript = [
        >>>     ('focus', 'x-terminal-emulator.X-terminal-emulator'),
        >>>     ('type', text),
        >>>     ('key', 'KP_Enter'),
        >>> ]
        >>> ut.util_ubuntu.XCtrl.do(*doscript, sleeptime=.01)


    """
    # @staticmethod
    # def send_raw_key_input(keys):
    #     import utool as ut
    #     print('send key input: %r' % (keys,))
    #     args = ['xdotool', 'type', keys]
    #     ut.cmd(*args, quiet=True, silence=True)

    @staticmethod
    def findall_window_ids(pattern):
        import utool as ut
        cmdkw = dict(verbose=False, quiet=True, silence=True)
        winid_list = ut.cmd(
            "wmctrl -lx | grep %s | awk '{print $1}'" % (pattern,),
            **cmdkw)[0].strip().split('\n')
        winid_list = [h for h in winid_list if h]
        return winid_list

    @staticmethod
    def find_window_id(pattern, method='mru'):
        import utool as ut
        cmdkw = dict(verbose=False, quiet=True, silence=True)
        winid_candidates = XCtrl.findall_window_ids(pattern)
        if len(winid_candidates) == 0:
            win_id = None
        elif len(winid_candidates) == 1:
            win_id = winid_candidates[0]
        else:
            if method == 'mru':
                # Find most recently used window with the focus name.
                winid_order_str = ut.cmd(
                    'xprop -root|grep "^_NET_CLIENT_LIST_STACKING"', **cmdkw)[0]
                winid_order = winid_order_str.split('#')[1].strip().split(', ')[::-1]
                winid_order_ = [int(h, 16) for h in winid_order]
                winid_candidates_ = [int(h, 16) for h in winid_candidates]
                win_id = hex(ut.isect(winid_order_, winid_candidates_)[0])
            else:
                raise NotImplementedError(method)
        return win_id

    @staticmethod
    def do(*cmd_list, **kwargs):
        import utool as ut
        import time
        verbose = kwargs.get('verbose', False)
        # print('Running xctrl.do script')
        if verbose:
            print('Executing x do: %s' % (ut.repr3(cmd_list),))

        cmdkw = dict(verbose=False, quiet=True, silence=True)
        # http://askubuntu.com/questions/455762/xbindkeys-wont-work-properly
        # Make things work even if other keys are pressed
        defaultsleep = 0.0
        sleeptime = kwargs.get('sleeptime', defaultsleep)
        time.sleep(.05)
        ut.cmd('xset r off', **cmdkw)
        memory = {}

        for item in cmd_list:
            # print('item = %r' % (item,))
            sleeptime = kwargs.get('sleeptime', defaultsleep)

            assert isinstance(item, tuple)
            assert len(item) >= 2
            xcmd, key_ = item[0:2]
            if len(item) >= 3:
                sleeptime = float(item[2])

            if xcmd == 'focus':
                key_ = str(key_)
                if key_.startswith('$'):
                    key_ = memory[key_[1:]]
                pattern = key_
                win_id = XCtrl.find_window_id(pattern, method='mru')
                if win_id is None:
                    args = ['wmctrl', '-xa', pattern]
                else:
                    args = ['wmctrl', '-ia', win_id]

            elif xcmd == 'focus_id':
                key_ = str(key_)
                if key_.startswith('$'):
                    key_ = memory[key_[1:]]
                args = ['wmctrl', '-ia', key_]
            elif xcmd == 'remember_window_id':
                out, err, ret = ut.cmd('xdotool getwindowfocus', **cmdkw)
                memory[key_] = out.strip()
                continue
            elif xcmd == 'remember_window_name':
                out, err, ret = ut.cmd('xdotool getwindowfocus getwindowname', **cmdkw)
                import pipes
                memory[key_] = pipes.quote(out.strip())
                continue
            elif xcmd == 'type':
                args = [
                    'xdotool',
                    'keyup', '--window', '0', '7',
                    'type', '--clearmodifiers',
                    '--window', '0', str(key_)
                ]
            else:
                args = ['xdotool', str(xcmd), str(key_)]

            if verbose:
                print('args = %r' % (args,))
                print('Exec: %s' % (' '.join(args),))
            # print('args = %r' % (args,))
            ut.cmd(*args, **cmdkw)
            if sleeptime > 0:
                time.sleep(sleeptime)

        ut.cmd('xset r on', verbose=False, quiet=True, silence=True)

    @staticmethod
    def focus_window(winhandle, path=None, name=None, sleeptime=.01):
        """
        sudo apt-get install xautomation
        apt-get install autokey-gtk

        wmctrl -xa gnome-terminal.Gnome-terminal
        wmctrl -xl
        """
        import utool as ut
        import time
        print('focus: ' + winhandle)
        args = ['wmctrl', '-xa', winhandle]
        ut.cmd(*args, verbose=False, quiet=True)
        time.sleep(sleeptime)


xctrl = XCtrl


def monitor_mouse():
    """
    CommandLine:
        python -m utool.util_ubuntu monitor_mouse

    Example:
        >>> # SCRIPT
        >>> from utool.util_ubuntu import *  # NOQA
        >>> import utool as ut
        >>> monitor_mouse()
    """
    import utool as ut
    import re
    import parse
    mouse_ids = ut.cmd('xinput --list ', verbose=False, quiet=True)[0]
    x = mouse_ids.decode('utf-8')
    pattern = 'mouse'
    pattern = 'trackball'
    print(x)
    grepres = ut.greplines(x.split('\n'), pattern, reflags=re.IGNORECASE)
    mouse_id = parse.parse('{left}id={id}{right}', grepres[0][0])['id']
    print('mouse_id = %r' % (mouse_id,))
    import time
    while True:
        time.sleep(.2)
        out = ut.cmd('xinput --query-state ' + mouse_id, verbose=False, quiet=True)[0]
        print(out)


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
