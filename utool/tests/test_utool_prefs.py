from __future__ import absolute_import, division, print_function
import guitool


def test_pref():
    guitool.ensure_qtapp()
    import utool
    root = utool.Pref(name='mypref')
    root.pref1 = 5
    root.pref2 = 'string'
    root.pref3 = 'True'
    root.pref4 = True
    root.pref5 = utool.Pref(name='child')
    root.pref5.pref6 = 3.5
    root.createQWidget()
    return root


if __name__ == '__main__':
    root = test_pref()
    guitool.qtapp_loop_nonblocking()
