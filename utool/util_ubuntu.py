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
        >>> # DISABLE_DOCTEST
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
    r"""
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
        echo $win_order

    CommandLine:
        python -m utool.util_ubuntu XCtrl

    Example:
        >>> # DISABLE_DOCTEST
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
    def move_window(win_key, bbox):
        """
        CommandLine:
            # List windows
            wmctrl -l
            # List desktops
            wmctrl -d

            # Window info
            xwininfo -id 60817412

            python -m utool.util_ubuntu XCtrl.move_window joncrall 0+1920,680,400,600,400
            python -m utool.util_ubuntu XCtrl.move_window joncrall [0,0,1000,1000]
            python -m utool.util_ubuntu XCtrl.move_window GVIM special2
            python -m utool.util_ubuntu XCtrl.move_window joncrall special2
            python -m utool.util_ubuntu XCtrl.move_window x-terminal-emulator.X-terminal-emulator [0,0,1000,1000]

        # >>> import utool as ut
        # >>> from utool import util_ubuntu
        # >>> orig_window = []
        # >>> X = util_ubuntu.XCtrl
        win_key =  'x-terminal-emulator.X-terminal-emulator'
        win_id = X.findall_window_ids(key)[0]

        python -m utool.util_ubuntu XCtrl.findall_window_ids gvim --src

        """
        import utool as ut
        try:
            import plottool_ibeis.screeninfo as screeninfo
        except ImportError:
            import plottool.screeninfo as screeninfo
        monitor_infos = {
            i + 1: screeninfo.get_resolution_info(i)
            for i in range(2)
        }
        # TODO: cut out borders
        # TODO: fix screeninfo monitor offsets
        # TODO: dynamic num screens
        def rel_to_abs_bbox(m, x, y, w, h):
            """ monitor_num, relative x, y, w, h """
            minfo = monitor_infos[m]
            # print('minfo(%d) = %s' % (m, ut.repr3(minfo),))
            mx, my = minfo['off_x'], minfo['off_y']
            mw, mh = minfo['pixels_w'], minfo['pixels_h']
            # Transform to the absolution position
            abs_x = (x * mw) + mx
            abs_y = (y * mh) + my
            abs_w = (w * mw)
            abs_h = (h * mh)
            abs_bbox = [abs_x, abs_y, abs_w, abs_h]
            abs_bbox = ','.join(map(str, map(int, abs_bbox)))
            return abs_bbox

        if win_key.startswith('joncrall') and bbox == 'special2':
            # Specify the relative position
            abs_bbox = rel_to_abs_bbox(m=2,
                                       x=0.0, y=0.7,
                                       w=1.0, h=0.3)
        elif win_key.startswith('GVIM') and bbox == 'special2':
            # Specify the relative position
            abs_bbox = rel_to_abs_bbox(m=2,
                                       x=0.0, y=0.0,
                                       w=1.0, h=0.7)
        else:
            abs_bbox = ','.join(map(str, eval(bbox)))

        print('MOVING: win_key = %r' % (win_key,))
        print('TO: abs_bbox = %r' % (abs_bbox,))
        # abs_bbox.replace('[', '').replace(']', '')
        # get = lambda cmd: ut.cmd2(' '.join(["/bin/bash", "-c", cmd]))['out']  # NOQA
        win_id = XCtrl.find_window_id(win_key, error='raise')
        print('MOVING: win_id = %r' % (win_id,))
        fmtdict = locals()
        cmd_list = [
            ("wmctrl -ir {win_id} -b remove,maximized_horz".format(**fmtdict)),
            ("wmctrl -ir {win_id} -b remove,maximized_vert".format(**fmtdict)),
            ("wmctrl -ir {win_id} -e 0,{abs_bbox}".format(**fmtdict)),
        ]
        print('\n'.join(cmd_list))
        for cmd in cmd_list:
            ut.cmd2(cmd)

    @staticmethod
    def findall_window_ids(pattern):
        """
        CommandLine:
            wmctrl  -l
            python -m utool.util_ubuntu XCtrl.findall_window_ids gvim --src
            python -m utool.util_ubuntu XCtrl.findall_window_ids gvim --src
            python -m utool.util_ubuntu XCtrl.findall_window_ids joncrall --src

        xprop -id

        wmctrl -l | awk '{print $1}' | xprop -id

        0x00a00007 | grep "WM_CLASS(STRING)"

        """
        import utool as ut
        cmdkw = dict(verbose=False, quiet=True, silence=True)
        command = "wmctrl -lx | grep '%s' | awk '{print $1}'" % (pattern,)
        # print(command)
        winid_list = ut.cmd(command, **cmdkw)[0].strip().split('\n')
        winid_list = [h for h in winid_list if h]
        winid_list = [int(h, 16) for h in winid_list]
        return winid_list

    @staticmethod
    def sort_window_ids(winid_list, order='mru'):
        """
        Orders window ids by most recently used
        """
        import utool as ut
        winid_order = XCtrl.sorted_window_ids(order)
        sorted_win_ids = ut.isect(winid_order, winid_list)
        return sorted_win_ids

    @staticmethod
    def killold(pattern, num=4):
        """
        Leaves no more than `num` instances of a program alive.  Ordering is
        determined by most recent usage.

        CommandLine:
            python -m utool.util_ubuntu XCtrl.killold gvim 2

        >>> import utool as ut
        >>> from utool import util_ubuntu
        >>> XCtrl = util_ubuntu.XCtrl
        >>> pattern = 'gvim'
        >>> num = 2

        """
        import utool as ut
        cmdkw = dict(verbose=False, quiet=True, silence=True)
        num = int(num)
        winid_list = XCtrl.findall_window_ids(pattern)
        winid_list = XCtrl.sort_window_ids(winid_list, 'mru')[num:]
        output_lines = ut.cmd(
            """wmctrl -lxp | awk '{print $1 " " $3}'""",
            **cmdkw)[0].strip().split('\n')
        output_fields = [line.split(' ') for line in output_lines]
        output_fields = [(int(wid, 16), int(pid)) for wid, pid in output_fields]
        pid_list = [pid for wid, pid in output_fields if wid in winid_list]
        import psutil
        for pid in pid_list:
            proc = psutil.Process(pid=pid)
            proc.kill()

    @staticmethod
    def sorted_window_ids(order='mru'):
        """
        Returns window ids orderd by criteria
        default is mru (most recently used)

        CommandLine:
            xprop -root | grep "^_NET_CLIENT_LIST_STACKING" | tr "," " "
            python -m utool.util_ubuntu XCtrl.sorted_window_ids
        """
        import utool as ut
        if order in ['mru', 'lru']:
            cmdkw = dict(verbose=False, quiet=True, silence=True)
            winid_order_str = ut.cmd(
                'xprop -root | grep "^_NET_CLIENT_LIST_STACKING"', **cmdkw)[0]
            winid_order = winid_order_str.split('#')[1].strip().split(', ')[::-1]
            winid_order = [int(h, 16) for h in winid_order]
            if order == 'lru':
                winid_order = winid_order[::-1]
        else:
            raise NotImplementedError(order)
        return winid_order

    @staticmethod
    def find_window_id(pattern, method='mru', error='raise'):
        """
        xprop -id 0x00a00007 | grep "WM_CLASS(STRING)"
        """
        import utool as ut
        winid_candidates = XCtrl.findall_window_ids(pattern)
        if len(winid_candidates) == 0:
            if error == 'raise':
                available_windows = ut.cmd2('wmctrl -l')['out']
                msg = 'No window matches pattern=%r' % (pattern,)
                msg += '\navailable windows are:\n%s' % (available_windows,)
                print(msg)
                raise Exception(msg)
            win_id = None
        elif len(winid_candidates) == 1:
            win_id = winid_candidates[0]
        else:
            # print('Multiple (%d) windows matches pattern=%r' % (
            #     len(winid_list), pattern,))
            # Find most recently used window with the focus name.
            win_id = XCtrl.sort_window_ids(winid_candidates, method)[0]
        return win_id

    @staticmethod
    def current_gvim_edit(op='e', fpath=''):
        r"""
        CommandLine:
            python -m utool.util_ubuntu XCtrl.current_gvim_edit sp ~/.bashrc
        """
        import utool as ut
        fpath = ut.unexpanduser(ut.truepath(fpath))
        # print('fpath = %r' % (fpath,))
        ut.copy_text_to_clipboard(fpath)
        # print(ut.get_clipboard())
        doscript = [
            ('focus', 'gvim'),
            ('key', 'Escape'),
            ('type2', ';' + op + ' ' + fpath),
            # ('type2', ';' + op + ' '),
            # ('key', 'ctrl+v'),
            ('key', 'KP_Enter'),
        ]
        XCtrl.do(*doscript, verbose=0, sleeptime=.001)

    @staticmethod
    def copy_gvim_to_terminal_script(text, return_to_win="1", verbose=0, sleeptime=.02):
        """
        import utool.util_ubuntu
        utool.util_ubuntu.XCtrl.copy_gvim_to_terminal_script('print("hi")', verbose=1)
        python -m utool.util_ubuntu XCtrl.copy_gvim_to_terminal_script "echo hi" 1 1

        If this doesn't work make sure pyperclip is installed and set to xsel

        print('foobar')
        echo hi
        """
        # Prepare to send text to xdotool
        import utool as ut
        import utool.util_ubuntu
        ut.copy_text_to_clipboard(text)

        if verbose:
            print('text = %r' % (text,))
            print(ut.get_clipboard())

        import re
        terminal_pattern = r'\|'.join([
            'terminal',
            re.escape('terminator.Terminator'),  # gtk3 terminator
            re.escape('x-terminal-emulator.X-terminal-emulator'),  # gtk2 terminator
        ])

        # Build xdtool script
        doscript = [
            ('remember_window_id', 'ACTIVE_WIN'),
            # ('focus', 'x-terminal-emulator.X-terminal-emulator'),
            ('focus', terminal_pattern),
            ('key', 'ctrl+shift+v'),
            ('key', 'KP_Enter'),
        ]
        if '\n' in text:
            # Press enter twice for multiline texts
            doscript += [
                ('key', 'KP_Enter'),
            ]

        if return_to_win == "1":
            doscript += [
                ('focus_id', '$ACTIVE_WIN'),
            ]
        # execute script
        # verbose = 1
        utool.util_ubuntu.XCtrl.do(*doscript, sleeptime=sleeptime, verbose=verbose)

    @staticmethod
    def do(*cmd_list, **kwargs):
        import utool as ut
        import time
        import six
        import sys
        verbose = kwargs.get('verbose', False)
        orig_print = globals()['print']
        print = ut.partial(orig_print, file=kwargs.get('file', sys.stdout))
        # print('Running xctrl.do script')
        if verbose:
            print('Executing x do: %s' % (ut.repr4(cmd_list),))
        debug = False

        cmdkw = dict(verbose=False, quiet=True, silence=True)
        # http://askubuntu.com/questions/455762/xbindkeys-wont-work-properly
        # Make things work even if other keys are pressed
        defaultsleep = 0.0
        sleeptime = kwargs.get('sleeptime', defaultsleep)
        time.sleep(.05)
        out, err, ret = ut.cmd('xset r off', **cmdkw)
        if debug:
            print('----------')
            print('xset r off')
            print('ret = %r' % (ret,))
            print('err = %r' % (err,))
            print('out = %r' % (out,))

        memory = {}

        tmpverbose = 0
        for count, item in enumerate(cmd_list):
            # print('item = %r' % (item,))
            sleeptime = kwargs.get('sleeptime', defaultsleep)
            if tmpverbose:
                print('moving on')
            tmpverbose = 0
            nocommand = 0

            assert isinstance(item, tuple)
            assert len(item) >= 2
            xcmd, key_ = item[0:2]
            if len(item) >= 3:
                if isinstance(item[2], six.string_types) and item[2].endswith('?'):
                    sleeptime = float(item[2][:-1])
                    tmpverbose = 1
                    print('special command sleep')
                    print('sleeptime = %r' % (sleeptime,))
                else:
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
                    args = ['wmctrl', '-ia', hex(win_id)]
            elif xcmd == 'focus_id':
                key_ = str(key_)
                if key_.startswith('$'):
                    key_ = memory[key_[1:]]
                args = ['wmctrl', '-ia', hex(key_)]
            elif xcmd == 'remember_window_id':
                out, err, ret = ut.cmd('xdotool getwindowfocus', **cmdkw)
                memory[key_] = int(out.strip())
                nocommand = True
                args = []
            elif xcmd == 'remember_window_name':
                out, err, ret = ut.cmd('xdotool getwindowfocus getwindowname', **cmdkw)
                import pipes
                memory[key_] = pipes.quote(out.strip())
                nocommand = True
                args = []
            elif xcmd == 'type':
                args = [
                    'xdotool',
                    'keyup', '--window', '0', '7',
                    'type', '--clearmodifiers',
                    '--window', '0', str(key_)
                ]
            elif xcmd == 'type2':
                import pipes
                args = [
                    'xdotool', 'type', pipes.quote(str(key_))
                ]
            elif xcmd == 'xset-r-on':
                args = ['xset', 'r', 'on']
            elif xcmd == 'xset-r-off':
                args = ['xset', 'r', 'off']
            else:
                args = ['xdotool', str(xcmd), str(key_)]

            if verbose or tmpverbose:
                print('\n\n# Step %d' % (count,))
                print(args, ' '.join(args))

            if nocommand:
                continue
                # print('args = %r -> %s' % (args, ' '.join(args),))
            # print('args = %r' % (args,))
            out, err, ret = ut.cmd(*args, **cmdkw)
            if debug:
                print('---- ' + xcmd + ' ------')
                print(' '.join(args))
                print('ret = %r' % (ret,))
                print('err = %r' % (err,))
                print('out = %r' % (out,))

            if sleeptime > 0:
                time.sleep(sleeptime)

        out, err, ret = ut.cmd('xset r on', verbose=False, quiet=True,
                               silence=True)
        if debug:
            print('----------')
            print('xset r on')
            print('ret = %r' % (ret,))
            print('err = %r' % (err,))
            print('out = %r' % (out,))

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
        >>> # DISABLE_DOCTEST
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
