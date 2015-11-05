# -*- coding: utf-8 -*-
import utool

#def ibeis_wc():


if __name__ == '__main__':
    dpath = '.'
    fpath_list = utool.glob(dpath, '*.py', recursive=True)
    def get_file_stats(fpath):
        text = utool.read_from(fpath, verbose=False)
        lc = len(text.splitlines())
        wc = len(text.split(' '))
        return lc, wc

    stat_list = [get_file_stats(fpath) for fpath in fpath_list]
    lc = sum([stat[0] for stat in stat_list])
    wc = sum([stat[1] for stat in stat_list])

    #wc = sum(len(utool.read_from(fpath).split(' ')) for fpath in fpath_list)
    print('word count = %r' % wc)
    print('line count = %r' % lc)
