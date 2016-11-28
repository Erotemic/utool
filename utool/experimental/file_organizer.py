# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import utool as ut
from os.path import join, basename, splitext, exists, dirname


class SourceDir(ut.NiceRepr):
    def __init__(self, dpath):
        self.dpath = dpath
        self.rel_fpath_list = None

    def populate(self):
        self.rel_fpath_list = ut.glob(self.dpath, '*',  recursive=True,
                                      fullpath=False, with_dirs=False)
        self.attrs = {
            'nbytes': list(map(ut.get_file_nBytes, self.fpaths())),
            'fname': list(map(basename, self.rel_fpath_list)),
            'ext': list(map(lambda p: splitext(p)[1].lower().replace('.jpeg', '.jpg'), self.rel_fpath_list)),
        }
        # self.nbytes_list = list(map(ut.get_file_nBytes, self.fpaths()))
        # self.fname_list = list(map(basename, self.rel_fpath_list))
        # self.ext_list = list(map(lambda p: splitext(p)[1].lower().replace('.jpeg', '.jpg'), self.rel_fpath_list))

    def __len__(self):
        return len(self.rel_fpath_list)

    def index(self):
        prog =  ut.ProgIter(self.fpaths(), length=len(self), label='building uuid')
        self.uuids = [ut.get_file_uuid(fpath) for fpath in prog]

    def duplicates(self):
        uuid_to_dupxs = ut.find_duplicate_items(self.uuids)
        dup_fpaths = [ut.take(self.rel_fpath_list, idxs) for idxs in uuid_to_dupxs.values()]
        return dup_fpaths

    def nbytes(self):
        return sum(self.attrs['nbytes'])

    def ext_hist(self):
        return ut.dict_hist(self.attrs['ext'])

    def fpaths(self):
        for fpath in self.rel_fpath_list:
            yield join(self.dpath, fpath)

    def __nice__(self):
        return self.dpath

    def isect_info(self, other):
        set1 = set(self.rel_fpath_list)
        set2 = set(other.rel_fpath_list)

        set_comparisons = ut.odict([
            ('union', set1.union(set2)),
            ('isect', set1.intersection(set2)),
            ('s1 - s2', set1.difference(set2)),
            ('s2 - s1', set1.difference(set1)),
        ])
        stat_stats = ut.map_vals(len, set_comparisons)
        print(ut.repr4(stat_stats))

        if False:
            idx_lookup1 = ut.make_index_lookup(self.rel_fpath_list)
            idx_lookup2 = ut.make_index_lookup(other.rel_fpath_list)

            uuids1 = ut.take(self.uuids, ut.take(idx_lookup1, set_comparisons['union']))
            uuids2 = ut.take(other.uuids, ut.take(idx_lookup2, set_comparisons['union']))

            uuids1 == uuids2

    def make_merge_bash_script(self, dest):
        import subprocess
        # find $SOURCE_DIR -name '*' -type f -exec mv -f {} $TARGET_DIR \;
        # bash_cmd = subprocess.list2cmdline(['mv', '--verbose', join(self.dpath, '*'), dest.dpath])
        bash_cmd = subprocess.list2cmdline(
            ['find', self.dpath, '-name', '\'*\'', '-type', 'f', '-exec', 'mv', '-f', '{}', dest.dpath, '\;'])
        print(bash_cmd)
        return bash_cmd

        # # import shutil
        # move_tasks = [
        #     (join(self.dpath, rel_fpath), join(dest.dpath, rel_fpath))
        #     for rel_fpath in self.rel_fpath_list
        # ]
        # for src, dst in move_tasks:
        #     if exists(dst):
        #         raise Exception('dont overwrite yet')

        # bash_script = '\n'.join([subprocess.list2cmdline(('mv', src, dst)) for src, dst in move_tasks])
        # return bash_script

    def merge_into(self, dest):
        import shutil
        print('Preparing merge %r into %r' % (self, dest))
        # import shutil
        move_tasks = [
            (join(self.dpath, rel_fpath), join(dest.dpath, rel_fpath))
            for rel_fpath in self.rel_fpath_list
        ]
        for src, dst in move_tasks:
            if exists(dst):
                raise Exception('dont overwrite yet')

        def trymove(src, dst):
            try:
                shutil.move(src, dst)
            except OSError:
                return 1
            return 0

        error_list = [
            trymove(src, dst)
            for (src, dst) in ut.ProgIter(move_tasks, lbl='moving')
        ]
        assert not any(error_list), 'error merging'
        return error_list

    def delete_empty_directories(self):
        """
        ut.ensuredir(self.dpath + '/foo')
        ut.ensuredir(self.dpath + '/foo/bar')
        ut.ensuredir(self.dpath + '/foo/bar/baz')
        self.delete_empty_directories()
        """
        import os
        # for root, dirs, files in os.walk(self.dpath, topdown=False):
        #     if len(files) == 0 and len(os.listdir(root)) == 0:
        #         print('Remove %s' % root)
        #         os.rmdir(root)

        if True:
            # Find all directories with no files
            subdirs = ut.glob(self.dpath, '*',  recursive=True, fullpath=False,
                              with_files=False, with_dirs=True)
            freq = {d: 0 for d in subdirs + ['']}
            for path in self.rel_fpath_list:
                while True:
                    path = dirname(path)
                    freq[path] += 1
                    if not path:
                        break
            to_delete = [k for k, v in freq.items() if v == 0]
            # Remove deep dirs first
            to_delete = ut.sortedby(to_delete, map(len, to_delete))[::-1]
            for d in to_delete:
                dpath = join(self.dpath, d)
                print('Remove %s' % dpath)
                os.rmdir(dpath)


def turtles():
    source_dpaths = sorted(ut.glob('/raid/raw/RotanTurtles/', '*',
                                   recusrive=False, with_dirs=True,
                                   with_files=False))
    sources = [SourceDir(dpath) for dpath in source_dpaths]

    for self in ut.ProgIter(sources, label='populate'):
        self.populate()

    import fnmatch
    del_ext = set(['.npy', '.flann', '.npz'])
    for self in ut.ProgIter(sources, label='populate'):
        flags = [ext in del_ext for ext in self.attrs['ext']]
        to_delete = ut.compress(list(self.fpaths()), flags)
        ut.remove_file_list(to_delete)
        flags = [fnmatch.fnmatch(fpath, '*/_hsdb/computed/chips/*.png') for fpath in self.rel_fpath_list]
        to_delete = ut.compress(list(self.fpaths()), flags)
        ut.remove_file_list(to_delete)
        self.populate()

    for self in ut.ProgIter(sources, label='del empty'):
        self.populate()
        self.delete_empty_directories()

    print(ut.byte_str2(sum([self.nbytes() for self in sources])))
    # [ut.byte_str2(self.nbytes()) for self in sources]

    # import numpy as np
    # num_isect = np.zeros((len(sources), len(sources)))
    # num_union = np.zeros((len(sources), len(sources)))

    for i, j in ut.combinations(range(len(sources)), 2):
        s1 = sources[i]
        s2 = sources[j]
        isect = set(s1.rel_fpath_list).intersection(s2.rel_fpath_list)
        # union = set(s1.rel_fpath_list).union(s2.rel_fpath_list)
        if isect:
            s1.isect_info(s2)
            print((i, j))
            print(s1.dpath)
            print(s2.dpath)
            self = s1
            other = s2
            assert False
            # print(isect)
            # break
        # num_isect[i, j] = len(isect)
        # num_union[i, j] = len(union)

    # for self in ut.ProgIter(sources, label='index'):
    #     self.index()

    for self in ut.ProgIter(sources, label='populate'):
        self.populate()

    dest = sources[0]
    others = sources[1:]
    # Merge others into dest
    bash_script = '\n'.join([other.make_merge_bash_script(dest) for other in others])
    print(bash_script)

    other = self

    for other in others:
        other.merge_into(dest)

    # [ut.byte_str2(self.nbytes()) for self in sources]

    # for self in sources:
    #     pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.experimental.file_organizer
        python -m utool.experimental.file_organizer --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
