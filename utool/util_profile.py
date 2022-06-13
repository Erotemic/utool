#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cleaning script for the output of utool profiling

Removes profiled output of code that never ran
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import range, cStringIO
import six
import operator
import utool as ut

"""
input_fname = 'raw_profile.dev.py.2014-09-23_18-28-57.raw.prof'
input_fname = 'raw_profile.qt_inc_automatch.py.2014-12-23_11-31-53.raw.prof'
"""


class Profiler(object):
    def __init__(self):
        import line_profiler
        self._lineprof = line_profiler.LineProfiler()

    def __call__(self, func):
        wrapper = self._lineprof(func)
        # wrapper = ut.preserve_sig(wrapper, func)
        return wrapper

    def get_output(self):
        file_ = cStringIO()
        self._lineprof.print_stats(stream=file_, stripzeros=True)
        file_.seek(0)
        text =  file_.read()
        output_text, summary_text = ut.clean_line_profile_text(text)
        return output_text


def make_profiler():
    import line_profiler
    profile = line_profiler.LineProfiler()
    return profile


def get_profile_text(profile):
    file_ = cStringIO()
    profile.print_stats(stream=file_, stripzeros=True)
    file_.seek(0)
    text =  file_.read()
    output_text, summary_text = ut.clean_line_profile_text(text)
    return output_text, summary_text


def dump_profile_text():
    import utool as ut
    print("Dumping Profile Information")
    profile = ut.PROFILE_FUNC
    try:
        output_text, summary_text = get_profile_text(profile)
    except AttributeError:
        print('profile is not on')
    else:
        #profile.dump_stats('out.lprof')
        print(summary_text)
        ut.writeto('profile_output.txt', output_text + '\n' + summary_text)
        ut.writeto('profile_output.%s.txt' % (ut.get_timestamp()),
                   output_text + '\n' + summary_text)


def __dbg_list(profile_block_list):
    for item in profile_block_list:
        newline = item.find('\n')
        twoline = item[newline + 1:].find('\n')
        head = item[0:(newline + twoline)]
        print(head)


def parse_rawprofile_blocks(text):
    """
    Split the file into blocks along delimters and and put delimeters back in
    the list
    """
    # The total time reported in the raw output is from pystone not kernprof
    # The pystone total time is actually the average time spent in the function
    delim = 'Total time: '
    delim2 = 'Pystone time: '
    #delim = 'File: '
    profile_block_list = ut.regex_split('^' + delim, text)
    for ix in range(1, len(profile_block_list)):
        profile_block_list[ix] = delim2 + profile_block_list[ix]
    return profile_block_list


def get_block_totaltime(block):
    time_line = ut.regex_search('Pystone time: [0-9.]* s', block)
    time_str  = ut.regex_search('[0-9.]+', time_line)
    if time_str is not None:
        return float(time_str)
    else:
        return None


def get_block_id(block):
    import re
    fpath_regex = ut.named_field('fpath', r'\S+')
    funcname_regex = ut.named_field('funcname', r'\S+')
    lineno_regex = ut.named_field('lineno', '[0-9]+')

    fileline_regex = 'File: ' + fpath_regex + '$'
    funcline_regex = 'Function: ' + funcname_regex + ' at line ' + lineno_regex + '$'
    fileline_match = re.search(fileline_regex, block, flags=re.MULTILINE)
    funcline_match = re.search(funcline_regex, block, flags=re.MULTILINE)
    if fileline_match is not None and funcline_match is not None:
        fpath    = fileline_match.groupdict()['fpath']
        funcname = funcline_match.groupdict()['funcname']
        lineno   = funcline_match.groupdict()['lineno']
        block_id = funcname + ':' + fpath + ':' + lineno
    else:
        block_id = 'None:None:None'
    return block_id


def parse_timemap_from_blocks(profile_block_list):
    """
    Build a map from times to line_profile blocks
    """
    prefix_list = []
    timemap = ut.ddict(list)
    for ix in range(len(profile_block_list)):
        block = profile_block_list[ix]
        total_time = get_block_totaltime(block)
        # Blocks without time go at the front of sorted output
        if total_time is None:
            prefix_list.append(block)
        # Blocks that are not run are not appended to output
        elif total_time != 0:
            timemap[total_time].append(block)
    return prefix_list, timemap


def get_summary(profile_block_list, maxlines=20):
    """
    References:
        https://github.com/rkern/line_profiler
    """
    time_list = [get_block_totaltime(block) for block in profile_block_list]
    time_list = [time if time is not None else -1 for time in time_list]
    blockid_list = [get_block_id(block) for block in profile_block_list]
    sortx = ut.list_argsort(time_list)
    sorted_time_list = ut.take(time_list, sortx)
    sorted_blockid_list = ut.take(blockid_list, sortx)

    aligned_blockid_list = ut.util_str.align_lines(sorted_blockid_list, ':')
    summary_lines = [('%6.2f seconds - ' % time) + line
                     for time, line in
                     zip(sorted_time_list, aligned_blockid_list)]
    #summary_header = ut.codeblock(
    #    '''
    #    CLEANED PROFILE OUPUT

    #    The Pystone timings are not from kernprof, so they may include kernprof
    #    overhead, whereas kernprof timings do not (unless the line being
    #    profiled is also decorated with kernrof)

    #    The kernprof times are reported in Timer Units

    #    ''')
    summary_lines_ = ut.listclip(summary_lines, maxlines, fromback=True)
    summary_text = '\n'.join(summary_lines_)
    return summary_text


def fix_rawprofile_blocks(profile_block_list):
    # TODO: finish function. should multiply times by
    # Timer unit to get true second profiling
    #profile_block_list_new = []
    for block in profile_block_list:
        block_lines = block.split('\n')
        sep = ['=' * 62]
        def split_block_at_sep(block_lines, sep):
            for pos, line in enumerate(block_lines):
                if line.find(sep) == 0:
                    pos += 1
                    header_lines = block_lines[:pos]
                    body_lines = block_lines[pos:]
                    return header_lines, body_lines
            return block_lines, None
        header_lines, body_lines = split_block_at_sep(block_lines, sep)


def clean_line_profile_text(text):
    """
    Sorts the output from line profile by execution time
    Removes entries which were not run
    """
    #
    profile_block_list = parse_rawprofile_blocks(text)
    #profile_block_list = fix_rawprofile_blocks(profile_block_list)
    #---
    # FIXME can be written much nicer
    prefix_list, timemap = parse_timemap_from_blocks(profile_block_list)
    # Sort the blocks by time
    sorted_lists = sorted(six.iteritems(timemap), key=operator.itemgetter(0))
    newlist = prefix_list[:]
    for key, val in sorted_lists:
        newlist.extend(val)
    # Rejoin output text
    output_text = '\n'.join(newlist)
    #---
    # Hack in a profile summary
    summary_text = get_summary(profile_block_list)
    output_text = output_text
    return output_text, summary_text


def clean_lprof_file(input_fname, output_fname=None):
    """ Reads a .lprof file and cleans it """
    # Read the raw .lprof text dump
    text = ut.read_from(input_fname)
    # Sort and clean the text
    output_text = clean_line_profile_text(text)
    return output_text


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_profile
        python -m utool.util_profile --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
