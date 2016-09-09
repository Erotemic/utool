# -*- coding: utf-8 -*-
"""
progress handler.

Old progress funcs needto be depricated ProgressIter and ProgChunks are pretty
much the only useful things here.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import time
import math
import datetime
from functools import partial
from utool import util_logging
from utool import util_inject
from utool import util_arg
from utool import util_time
from utool import util_iter
from utool import util_cplat
from six.moves import range, zip
import collections
import six  # NOQA
print, rrr, profile = util_inject.inject2(__name__, '[progress]')

default_timer = util_time.default_timer


QUIET = util_arg.QUIET
SILENT = util_arg.SILENT
VALID_PROGRESS_TYPES = ['none', 'dots', 'fmtstr', 'simple']
AGGROFLUSH = util_arg.get_argflag('--aggroflush')
PROGGRESS_BACKSPACE = not util_arg.get_argflag(('--screen', '--progress-backspace'))
NO_PROGRESS = util_arg.get_argflag(('--no-progress', '--noprogress'))
FORCE_ALL_PROGRESS = util_arg.get_argflag(('--force-all-progress',))
#('--screen' not in sys.argv and '--progress-backspace' not in sys.argv)

DEBUG_FREQ_ADJUST = util_arg.get_argflag('--debug-adjust-freq')


#PROGRESS_WRITE = sys.stdout.write
#PROGRESS_FLUSH = sys.stdout.flush

# FIXME: if this is loaded before logging beings
# progress will not be logged
#PROGRESS_WRITE = util_logging.__UTOOL_WRITE__
PROGRESS_WRITE = print
PROGRESS_FLUSH = util_logging.__UTOOL_FLUSH__


def test_progress():
    """
    CommandLine:
        python -m utool.util_progress --test-test_progress

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_progress import *  # NOQA
        >>> test_progress()
    """
    import utool as ut
    #import time
    #ut.rrrr()
    print('_________________')
    #numiter = 50
    #sleeptime = 1E-4
    #sleeptime2 = 1E-2
    numiter = 20
    sleeptime = 1E-7
    sleeptime2 = 1E-7
    with ut.Timer():
        for x in ut.ProgressIter(range(0, numiter), freq=8, adjust=True):
            time.sleep(sleeptime)
    print('_________________')
    numiter = 50
    sleeptime = 1E-4
    with ut.Timer():
        for x in ut.ProgressIter(range(0, numiter), freq=8, adjust=True):
            time.sleep(sleeptime)
    print('_________________')
    print('No frequncy run:')
    with ut.Timer():
        for x in range(0, numiter):
            time.sleep(sleeptime)
    print('_________________')
    numiter = 500
    sleeptime = 8E-7
    with ut.Timer():
        for x in ut.ProgressIter(range(0, numiter), freq=8, adjust=True):
            time.sleep(sleeptime)
    print('_________________')
    with ut.Timer():
        for x in ut.ProgressIter(range(0, numiter), freq=200):
            time.sleep(sleeptime)
    print('_________________')
    print('No frequncy run:')
    with ut.Timer():
        for x in range(0, numiter):
            time.sleep(sleeptime)
    print('_________________')
    # Test nested iter
    # progiter1 = ut.ProgressIter(range(0, 10), lbl='prog1', freq=1, adjust=False)
    # for count1 in progiter1:
    #     progiter_partials = progiter1.get_subindexers(1)
    #     progiter2 = progiter_partials[0](range(0, 7), lbl='sub_prog1', freq=1, adjust=False)
    #     for count2 in progiter2:
    #         pass
    for x in ut.ProgressIter(zip(range(10), range(10)), freq=8, adjust=True):
        time.sleep(sleeptime)
        #progiter3 = progiter_partials[1](range(0, 3), lbl='sub_prog2', freq=1, adjust=False)
        #for count3 in progiter3:
        #    pass
    print('Double backspace progress 1')
    progiter1 = ut.ProgressIter(range(0, 10), lbl='prog1', freq=1, adjust=False, backspace=False)
    for count1 in progiter1:
        progiter2 = ut.ProgressIter(range(0, 10), lbl='prog2', freq=1, adjust=False, backspace=True)
        for count2 in progiter2:
            time.sleep(sleeptime2)

    print('Double backspace progress 2')
    progiter1 = ut.ProgressIter(range(0, 10), lbl='prog1', freq=1, adjust=False, backspace=True)
    for count1 in progiter1:
        progiter2 = ut.ProgressIter(range(0, 10), lbl='prog2', freq=1, adjust=False, backspace=True)
        for count2 in progiter2:
            time.sleep(sleeptime2)


def get_nTotalChunks(nTotal, chunksize):
    """
    Returns the number of chunks that a list will be split into given a
    chunksize.

    Args:
        nTotal (int):
        chunksize (int):

    Returns:
        int: nTotalChunks

    SeeAlso    from ibeis.algo.hots import graph_iden
    infr_list = []
    for annots in ut.ProgIter(flagged_annots, lbl='creating inference', freq=1, bs=True):
        aids = annots.aids
        nids = [1] * len(aids)
        infr = graph_iden.AnnotInference(ibs, aids, nids, verbose=False)
        infr.initialize_graph()
        infr.reset_feedback()
        infr.apply_feedback()
        infr_list.append(infr)

    for infr in ut.ProgIter(infr_list, lbl='flagging speeding edges', freq=1, bs=True):
        annots = ibs.annots(infr.aids)
        edge_to_speeds = annots.get_speeds()
        for (aid1, aid2), speed in edge_to_speeds.items():
            if speed > MAX_SPEED:
                if infr.graph.has_edge(aid1, aid2):
                    pass
                infr.add_feedback(aid1, aid2, 'nomatch')
        infr.apply_feedback()

    CommandLine:
        python -m utool.util_progress --exec-get_nTotalChunks:0
        python -m utool.util_progress --exec-get_nTotalChunks:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_progress import *  # NOQA
        >>> nTotal = 2000
        >>> chunksize = 256
        >>> nTotalChunks = get_nTotalChunks(nTotal, chunksize)
        >>> result = ('nTotalChunks = %s' % (six.text_type(nTotalChunks),))
        >>> print(result)
        nTotalChunks = 8

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_progress import *  # NOQA
        >>> nTotal = 256
        >>> chunksize = 32
        >>> nTotalChunks = get_nTotalChunks(nTotal, chunksize)
        >>> result = ('nTotalChunks = %s' % (six.text_type(nTotalChunks),))
        >>> print(result)
        nTotalChunks = 8
    """
    nTotalChunks = int(math.ceil(nTotal / chunksize))
    return nTotalChunks


def ProgChunks(list_, chunksize, nInput=None, **kwargs):
    """
    Yeilds an iterator in chunks and computes progress
    Progress version of ut.ichunks

    Args:
        list_ (list):
        chunksize (?):
        nInput (None): (default = None)

    Kwargs:
        nTotal, freq

    Returns:
        ProgressIter: progiter_

    CommandLine:
        python -m utool.util_progress ProgChunks --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_progress import *  # NOQA
        >>> import utool as ut
        >>> list_ = range(100)
        >>> chunksize = 10
        >>> nInput = None
        >>> progiter_ = ProgChunks(list_, chunksize, nInput)
        >>> iter_ = iter(progiter_)
        >>> chunk = six.next(iter_)
        >>> assert len(chunk) == 10
        >>> rest = ut.flatten(list(progiter_))
        >>> assert len(rest) == 90
    """
    if nInput is None:
        nInput = len(list_)
    nTotalChunks = get_nTotalChunks(nInput, chunksize)
    kwargs['nTotal'] = nTotalChunks
    if 'freq' not in kwargs:
        kwargs['freq'] = 1
    chunk_iter = util_iter.ichunks(list_, chunksize)
    progiter_ = ProgressIter(chunk_iter, **kwargs)
    return progiter_


def ProgPartial(*args, **kwargs):
    return partial(ProgressIter, *args, **kwargs)


class ProgressIter(object):
    """
    Wraps a for loop with progress reporting

    lbl='Progress: ', nTotal=0, flushfreq=4, startafter=-1, start=True,
    repl=False, approx=False, disable=False, writefreq=1, with_time=False,
    backspace=True, pad_stdout=False, wfreq=None, ffreq=None, freq=None,
    total=None, num=None, with_totaltime=None

    Referencs:
        https://github.com/verigak/progress/blob/master/progress/__init__.py

    Args:
        iterable (): iterable normally passed to for loop
        lbl (str):  progress label
        nTotal (int):
        flushfreq (int):
        startafter (int):
        start (bool):
        repl (bool):
        approx (bool):
        enabled (bool):
        writefreq (int):
        with_totaltime (bool):
        backspace (bool):
        pad_stdout (bool):
        autoadjust (bool): no adjusting frequency if True (default False)
        wfreq (None): alias for write_freq
        ffreq (None): alias for flush_freq
        total (None): alias for nTotal
        num (None):   alias for nTotal

    Timeit::
        import utool as ut
        setup = ut.codeblock(
        '''
        import utool as ut
        from six.moves import range, zip
        import time
        def time_append(size):
            start_time    = time.time()
            last_time     = start_time
            list2 = []
            for x in range(size):
                now_time    = time.time()
                between = now_time - last_time
                last_time   = now_time
                list2.append(between)

        def time_assign(size):
            start_time    = time.time()
            last_time     = start_time
            list1 = ut.alloc_nones(size)
            for x in range(size):
                now_time    = time.time()
                between = now_time - last_time
                last_time   = now_time
                list1[x] = between

        def time_baseline(size):
            start_time    = time.time()
            last_time     = start_time
            for x in range(size):
                now_time    = time.time()
                between = now_time - last_time
                last_time   = now_time

        def time_null(size):
            for x in range(size):
                pass
        ''')

        input_sizes = [2 ** count for count in range(7, 12)]
        stmt_list = ['time_assign', 'time_append', 'time_baseline', 'time_null']
        input_sizes=[100, 1000, 10000]
        ut.timeit_grid(stmt_list, setup, input_sizes=input_sizes, show=True)

    CommandLine:
        python -m utool.util_progress --test-ProgressIter
        python -m utool.util_progress --test-ProgressIter:0
        python -m utool.util_progress --test-ProgressIter:1
        python -m utool.util_progress --test-ProgressIter:2
        python -m utool.util_progress --test-ProgressIter:3


    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from six.moves import range
        >>> num = 1000
        >>> num2 = 10001
        >>> results1 = [x for x in ut.ProgressIter(range(num), wfreq=10, adjust=True)]
        >>> results4 = [x for x in ut.ProgressIter(range(num), wfreq=1, adjust=True)]
        >>> results2 = [x for x in range(num)]
        >>> results3 = [x for x in ut.progiter((y + 1 for y in range(num2)),
        >>>                                    nTotal=num2, wfreq=1000,
        >>>                                    backspace=True, adjust=True)]
        >>> assert results1 == results2

    Example1:
        >>> # SLOW_DOCTEST
        >>> import utool as ut
        >>> from six.moves import range
        >>> num2 = 10001
        >>> progiter = ut.ProgressIter(range(num2), lbl='testing primes',
        >>>                            report_unit='seconds', freq=1,
        >>>                            time_thresh=.1, adjust=True)
        >>> results1 = [ut.get_nth_prime_bruteforce(29) for x in progiter]
        >>> print(ut.truncate_str(ut.repr2(results1), 70))

    Example2:
        >>> # SLOW_DOCTEST
        >>> import utool as ut
        >>> from six.moves import range
        >>> num2 = 100001
        >>> progiter = ut.ProgressIter(range(num2), lbl='testing primes',
        >>>                            report_unit='seconds', freq=1,
        >>>                            time_thresh=3, adjust=True)
        >>> results1 = [ut.get_nth_prime_bruteforce(29) for x in progiter]
        >>> print(ut.truncate_str(ut.repr2(results1), 70))

    Example3:
        >>> # SLOW_DOCTEST
        >>> import utool as ut
        >>> from six.moves import range
        >>> import time
        >>> crazy_time_list = [.001, .01, .0001] * 1000
        >>> crazy_time_iter = (time.sleep(x) for x in crazy_time_list)
        >>> progiter = ut.ProgressIter(crazy_time_iter, lbl='crazy times', nTotal=len(crazy_time_list), freq=10)
        >>> list(progiter)

    """
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        if len(args) < 2 and 'nTotal' not in kwargs:
            try:
                nTotal = len(iterable)
                kwargs['nTotal'] = nTotal
            except Exception:
                pass
        self.use_rate           = kwargs.pop('use_rate', True)
        self.use_rate = True  # Force
        self.lbl                = kwargs.get('lbl', 'lbl')
        self.nTotal             = kwargs.get('nTotal', 0)
        #self.backspace          = kwargs.get('backspace', True)
        self.backspace          = kwargs.get('backspace', kwargs.get('bs', False))
        self.freq               = kwargs.get('freq', 1)
        self.invert_rate        = kwargs.get('invert_rate', False)
        #self.report_unit       = kwargs.get('report_unit', 'minutes')
        self.enabled            = kwargs.get('enabled', True)
        self.report_unit        = kwargs.get('report_unit', 'seconds')
        # autoadjust frequency of reporting
        self.autoadjust         = kwargs.get('autoadjust', kwargs.get('adjust', False))
        self.time_thresh        = kwargs.pop('time_thresh', None)
        self.prog_hook          = kwargs.pop('prog_hook', None)
        self.separate           = kwargs.pop('separate', False)

        # FIXME: get these subinder things working
        # ~/code/guitool/guitool/guitool_components.py
        #self.substep_min        = kwargs.pop('substep_min', 0)
        #self.substep_size       = kwargs.pop('substep_size', 1)
        #self.level              = kwargs.pop('level', 0)

        self.parent_index       = kwargs.pop('parent_index', 0)
        self.parent_nTotal      = kwargs.pop('parent_nTotal', 1)
        self.parent_offset      = self.parent_index * self.nTotal

        # Window sizes for estimates
        self.est_window         = kwargs.pop('est_window', 64)
        #self.start_offset       = self.substep_min

        if FORCE_ALL_PROGRESS:
            self.freq = 1
            self.autoadjust = False

        if self.prog_hook is not None:
            # Sets the label of a progress bar to the ProgIter label
            self.prog_hook.register_progiter(self)
        #self.time_thresh_growth = kwargs.pop('time_thresh_growth', 1.0)
        self.time_thresh_growth = kwargs.pop('time_thresh_growth', 1.0)
        self.with_totaltime = False
        if self.freq is None:
            self.freq = 1
        if self.use_rate:
            # Hacky so hacky. this needs major cleanup
            # saving args and kwargs so can wait on log_progress call
            # not sure where it is called and dont want to break things
            self.args = args
            self.kwargs = kwargs
            self.mark = None
            self.end = None
        #else:
        #    self.mark, self.end = log_progress(*args, **kwargs)
        self.count = 0

    def __call__(self, iterable):
        self.iterable = iterable
        return self

    def __iter__(self):
        if not self.enabled:
            return iter(self.iterable)
        if NO_PROGRESS:
            # IF PROGRESS IS TURNED OFF
            msg = 'Iterating ' + self.lbl + ' with no progress'
            if not util_arg.QUIET:
                print(msg)
            #with ut.Timer(msg):
            return iter(self.iterable)
        else:
            #if self.use_rate:
            # STANDARD CALL CASE
            return self.iter_rate()
            #else:
            #    return self.iter_without_rate()

    #def get_subindexers(prog_iter, num_substeps):
    #    # FIXME and  make this a method of progiter
    #    step_min = (((prog_iter.count - 1) / prog_iter.nTotal) *
    #                prog_iter.substep_size + prog_iter.substep_min)
    #    step_size = (1.0 / prog_iter.nTotal) * prog_iter.substep_size

    #    substep_size = step_size / num_substeps
    #    substep_min_list = [(step * substep_size) + step_min
    #                        for step in range(num_substeps)]
    #    #level = prog_iter.level + 1
    #    DEBUG = False
    #    if DEBUG:
    #        with ut.Indenter(' ' * 4 * prog_iter.level):
    #            print('\n')
    #            print('+____<NEW SUBSTEPS>____')
    #            print('Making %d substeps for prog_iter.lbl = %s' % (
    #                num_substeps, prog_iter.lbl,))
    #            print(' * step_min         = %.2f' % (step_min,))
    #            print(' * step_size        = %.2f' % (step_size,))
    #            print(' * substep_size     = %.2f' % (substep_size,))
    #            print(' * substep_min_list = %r' % (substep_min_list,))
    #            print(r'L____</NEW SUBSTEPS>____')
    #            print('\n')
    #    subprog_partial_list = [
    #        partial(ProgressIter,
    #                parent_nTotal=prog_iter.nTotal * num_substeps,
    #                parent_index=(prog_iter.count - 1) + (prog_iter.nTotal * step))
    #        for step in range(num_substeps)]
    #    return subprog_partial_list

    #def build_msg_fmtstr_time(self, lbl, invert_rate, backspace):
    #    with_wall = True
    #    tzname = time.tzname[0]
    #    if util_cplat.WIN32:
    #        tzname = tzname.replace('Eastern Standard Time', 'EST')
    #    msg_fmtstr_time = ''.join((
    #        'rate=%3.3f seconds/iter, ' if invert_rate else 'rate=%4.2f Hz,',
    #        ' etr=%s,',
    #        ' ellapsed=%s,',
    #        ' wall=%s ' + tzname if with_wall else '',
    #        #'' if backspace else '\n',
    #        '\n' if backspace else '',
    #    ))
    #    return msg_fmtstr_time

    #def build_msg_fmtstr(self, nTotal, lbl, invert_rate, backspace):
    #    msg_fmtstr = (self.build_msg_fmtstr_index(nTotal, lbl) +
    #                  self.build_msg_fmtstr_time(lbl, invert_rate, backspace))
    #    return msg_fmtstr

    @staticmethod
    def build_msg_fmtstr_head_cols(nTotal, lbl, backspace=True):
        nTotal_ = '?' if nTotal == 0 else six.text_type(nTotal)
        if backspace:
            #msg_head_columns = ['\r', lbl, ' {count:4d}/', nTotal_ , '...  ']
            msg_head_columns = ['', lbl, ' {count:4d}/', nTotal_ , '...  ']
        else:
            msg_head_columns = ['', lbl, ' {count:4d}/', nTotal_ , '...  ']
        return msg_head_columns

    @staticmethod
    def build_msg_fmtstr2(lbl, nTotal, invert_rate, backspace):
        r"""
        Args:
            lbl (str):
            invert_rate (bool):
            backspace (bool):

        Returns:
            str: msg_fmtstr_time

        CommandLine:
            python -m utool.util_progress --exec-ProgressIter.build_msg_fmtstr2

        Setup:
            >>> from utool.util_progress import *  # NOQA
            >>> lbl = 'foo'
            >>> invert_rate = True
            >>> backspace = False
            >>> nTotal = None

        Example:
            >>> # DISABLE_DOCTEST
            >>> msg_fmtstr_time = ProgressIter.build_msg_fmtstr2(lbl, nTotal, invert_rate, backspace)
            >>> result = ('%s' % (ut.repr2(msg_fmtstr_time),))
            >>> print(result)
        """
        with_wall = True
        tzname = time.tzname[0]
        if util_cplat.WIN32:
            tzname = tzname.replace('Eastern Standard Time', 'EST')
        msg_head = ProgressIter.build_msg_fmtstr_head_cols(nTotal, lbl, backspace)
        msg_tail = [
            ('rate={rate:3.3f} seconds/iter, '
             if invert_rate else 'rate={rate:4.2f} Hz,'),
            ('' if nTotal == 0 else ' etr={etr},'),
            ' ellapsed={ellapsed},',
            (' wall={wall} ' + tzname if with_wall else ''),
            #'' if backspace else '\n',
            # backslash-r is a carrage return and undoes all previous output on
            # a written line
            #'\r' if backspace else '\n',
            #'\r' if backspace else '\n',
            '\r' if backspace else '\n',
            #'' if backspace else '\n',
            #'\n' if backspace else '',
        ]
        msg_fmtstr_time = ''.join((msg_head + msg_tail))
        return msg_fmtstr_time

    def iter_rate(self):
        """
        pun not intended

        # TODO: record iteration times for analysis
        # TODO Incorporate this better
        # FIXME; pad_stdout into subfunctions

        import dis
        dis.dis(ut.ProgressIter.iter_rate)
        """
        #class IterState(object):
        #    def __init__(state):
        #        state.freq = 1
        #        state.freq = 1
        #        pass
        adjust = self.autoadjust
        # SETUP VARIABLES
        # HACK: reaquire logging print funcs in case they have changed
        PROGRESS_WRITE = util_logging.__UTOOL_WRITE__
        PROGRESS_FLUSH = util_logging.__UTOOL_FLUSH__
        #import sys
        #PROGRESS_WRITE = sys.stdout.write
        #PROGRESS_FLUSH = sys.stdout.flush

        nTotal        = self.nTotal * self.parent_nTotal  # hack
        freq          = self.freq
        self.count    = 0
        between_count = 0
        last_count    = 0

        # how long iterations should be before a flush
        # (used for freq adjustment)
        time_thresh = (self._get_timethresh_heuristics()
                       if self.time_thresh is None else
                       self.time_thresh)
        time_thresh_growth = self.time_thresh_growth
        if time_thresh_growth > 1:
            # time_thresh_growth is specified for very long processes
            # print out the starting timestamp in that case
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + time.tzname[0]
            print('Start progress lbl= %s at %s' % (self.lbl, timestamp,))
        #time_thresh = 0.5
        max_between_time = -1.0
        max_between_count = -1.0  # why is this different? # because frequency varies

        # TODO: should be kept as a statistic that uses the max time from a
        # list of iterations divided by the size of that list that will account
        # for buffering issues
        iters_per_second = -1

        #est_timeunit_left = -1

        # Write initial message
        #force_newlines = not self.backspace
        print_sep = self.separate
        if print_sep:
            print('---------')
        start_msg_fmt = ''.join(self.build_msg_fmtstr_head_cols(nTotal, self.lbl, backspace=self.backspace))
        # Prepare for iteration
        msg_fmtstr = self.build_msg_fmtstr2(self.lbl, nTotal,
                                            self.invert_rate, self.backspace)
        #msg_fmtstr = self.build_msg_fmtstr(nTotal, self.lbl,
        #                                   self.invert_rate, self.backspace)

        try:
            PROGRESS_FLUSH()
        except IOError as ex:
            # There is some weird error when doing progress in IPython notebook
            if util_arg.VERBOSE:
                print('IOError flushing %s' % (ex,))
        #PROGRESS_WRITE(start_msg)
        if self.backspace:
            msg = msg_fmtstr.format(
                count=self.count, rate=0.0,
                etr=six.text_type('0:00:00'),
                ellapsed=six.text_type('0:00:00'),
                wall=time.strftime('%H:%M'),
            )
            PROGRESS_WRITE(msg)
        else:
            start_msg = start_msg_fmt.format(count=self.parent_offset)
            PROGRESS_WRITE(start_msg + '\n')
        #PROGRESS_WRITE(self.build_msg_fmtstr_index(nTotal, self.lbl) % (self.parent_offset))
        #if force_newlines:
        #    PROGRESS_WRITE('\n')

        try:
            PROGRESS_FLUSH()
        except IOError as ex:
            # There is some weird error when doing progress in IPython notebook
            if util_arg.VERBOSE:
                print('IOError flushing %s' % (ex,))
            #print('PROGRESS_FLUSH = %r' % (PROGRESS_FLUSH,))
            #import utool as ut
            #ut.debug_logging_iostreams()
            #ut.printex(ex)
            #raise
        if self.prog_hook is not None:
            self.prog_hook(self.count, nTotal)

        # TODO: on windows is time.clock better?
        # http://exnumerus.blogspot.com/2011/02/how-to-quickly-plot-multiple-line.html
        start_time    = default_timer()
        last_time     = start_time

        start = 1 + self.parent_offset

        USE_RECORD = True
        # use last 64 times to compute a more stable average rate
        measure_between_time = collections.deque([], maxlen=self.est_window)
        measure_est_seconds = collections.deque([], maxlen=self.est_window)

        # Wrap the for loop with a generator
        for self.count, item in enumerate(self.iterable, start=start):
            # GENERATE
            yield item
            # DO PROGRESS INFO
            if (self.count) % freq == 0:
                # UPDATE INFO
                now_time          = default_timer()
                between_time      = (now_time - last_time)
                between_count     = self.count - last_count
                total_seconds     = (now_time - start_time)
                if USE_RECORD:
                    measure_between_time.append(between_count / (float(between_time) + 1E-9))
                    iters_per_second = sum(measure_between_time) / len(measure_between_time)
                else:
                    iters_per_second = between_count / (float(between_time) + 1E-9)
                # If the future is known
                if nTotal is None:
                    est_seconds_left = -1
                else:
                    iters_left = nTotal - self.count
                    est_etr = iters_left / (iters_per_second + 1E-9)
                    if USE_RECORD:
                        measure_est_seconds.append(est_etr)
                        est_seconds_left = sum(measure_est_seconds) / len(measure_est_seconds)
                    else:
                        est_seconds_left = est_etr

                # /future
                last_count        = self.count
                last_time         = now_time
                # ADJUST FREQ IF NEEDED
                # Adjust frequency if printing too quickly
                # so progress doesnt slow down actual function
                # TODO: better adjust algorithm
                time_thresh *= time_thresh_growth
                if adjust and (between_time < time_thresh or between_time > time_thresh * 2.0):
                    max_between_time = max(max(max_between_time, between_time), 1E-9)
                    max_between_count = max(max_between_count, between_count)
                    # If progress was uniform and all time estimates were
                    # perfect this would be the new freq to achieve time_thresh
                    new_freq = max(int(time_thresh * max_between_count / max_between_time), 1)
                    if DEBUG_FREQ_ADJUST:
                        print('\n+---')
                        print('[prog] between_count = %r' % between_count)
                        print('[prog] between_time = %.8r' % between_time)
                        print('[prog] time_thresh = %r' % time_thresh)
                        print('[prog] max_between_count = %r' % max_between_count)
                        print('[prog] max_between_time = %.8r' % max_between_time)
                        print('[prog] Adusting frequency from: %r' % freq)
                        print('[prog] Adusting frequency to: %r' % new_freq)
                        print('L___')
                    # But things are not perfect. So, don't make drastic changes
                    max_freq_change_up = max(256, freq * 2)
                    max_freq_change_down = freq // 2
                    if (new_freq - freq) > max_freq_change_up:
                        freq += max_freq_change_up
                    elif (freq - new_freq) > max_freq_change_down:
                        freq -= max_freq_change_down
                    else:
                        freq = new_freq
                #msg = msg_fmtstr % (
                #    self.count,
                #    1.0 / iters_per_second if self.invert_rate else iters_per_second,
                #    six.text_type(datetime.timedelta(seconds=int(est_seconds_left))),
                #    six.text_type(datetime.timedelta(seconds=int(total_seconds))),
                #    time.strftime('%H:%M'),
                #)
                msg = msg_fmtstr.format(
                    count=self.count,
                    rate=1.0 / iters_per_second if self.invert_rate else iters_per_second,
                    etr=six.text_type(datetime.timedelta(seconds=int(est_seconds_left))),
                    ellapsed=six.text_type(datetime.timedelta(seconds=int(total_seconds))),
                    wall=time.strftime('%H:%M'),
                )
                #est_timeunit_left,
                #total_timeunit)
                if print_sep:
                    print('---------')
                PROGRESS_WRITE(msg)
                #if force_newlines:
                #    PROGRESS_WRITE('\n')
                #if not self.backspace:
                try:
                    PROGRESS_FLUSH()
                except IOError as ex:
                    if util_arg.VERBOSE:
                        print('IOError flushing %s' % (ex,))
                # DO PROGRESS INFO
                if self.prog_hook is not None:
                    # From the point of view of the progress iter, we are
                    # about to enter the body of a for loop. (But we may have executed
                    # the body implicitly in the yeild....
                    # so it is ambiguous. In the second case 0 will be executed twice.
                    self.prog_hook(self.count, nTotal)
        # --- end of main loop
        # cleanup
        if (self.count) % freq != 0:
            # If the final line of progress was not written in the loop, write it here
            est_seconds_left = 0
            now_time = default_timer()
            total_seconds = (now_time - start_time)
            #msg = msg_fmtstr % (
            #    self.count, 1.0 / iters_per_second if self.invert_rate else iters_per_second,
            #    six.text_type(datetime.timedelta(seconds=int(est_seconds_left))),
            #    six.text_type(datetime.timedelta(seconds=int(total_seconds))),
            #    time.strftime('%H:%M'),
            #)
            msg = msg_fmtstr.format(
                count=self.count,
                rate=1.0 / iters_per_second if self.invert_rate else iters_per_second,
                etr=six.text_type(datetime.timedelta(seconds=int(est_seconds_left))),
                ellapsed=six.text_type(datetime.timedelta(seconds=int(total_seconds))),
                wall=time.strftime('%H:%M'),
            )
            PROGRESS_WRITE(msg)
            #if not self.backspace:
            try:
                PROGRESS_FLUSH()
            except IOError as ex:
                if util_arg.VERBOSE:
                    print('IOError flushing %s' % (ex,))
            #PROGRESS_WRITE('\nComplete(2)\n')
            if self.prog_hook is not None:
                # From the point of view of the progress iter, we are
                # about to enter the body of a for loop. (But we may have executed
                # the body implicitly in the yeild....
                # so it is ambiguous. In the second case 0 will be executed twice.
                self.prog_hook(self.count, nTotal)
        if not self.backspace:
            PROGRESS_WRITE('\n')
        else:
            PROGRESS_WRITE('\n')
            #PROGRESS_WRITE('\nComplete(1)\n')
        #import sys
        #sys.stdout.write('\nCOMPLETE@@@\n')
        #print('COMPLETE45')
        #cumrate = 1E-9
        #self.nTotal = len(self.iterable)
        # for:
        #    if:
        #         if:
        #            print('')
        #            freq = max(int(1.3 * between_count * time_thresh / between_time), 1)
        #            # There has to be a standard way to do this.
        #            # Refer to: https://github.com/verigak/progress/blob/master/progress/__init__.py
        #            freq = max(int((between_count * between_time) / time_thresh), 1)
        #            freq = max(int((between_count) / time_thresh), 1)
        #            freq = max(int((between_time) / time_thresh), 1)
        #            freq = max(int(time_thresh / between_count), 1)
        #            print('[prog] Adusting frequency to: %r' % freq)
        #            print('')
        #        cumrate += between_time
        #        rate = (self.count + 1.0) / float(cumrate)
        #        if False and __debug__:
        #            print('<!!!!!!!!!!!!!>')
        #            print('iters_left = %r' % iters_left)
        #            print('between_time = %r' % between_time)
        #            print('between_count = %r' % between_count)
        #            print('est_seconds_left = %r' % est_seconds_left)
        #            print('iters_per_second = %r' % iters_per_second)
        #            print('</!!!!!!!!!!!!!>')
        #PROGRESS_WRITE('\n')
        #msg = msg_fmtstr % (self.count + 1, iters_per_second, est_timeunit_left)
        #print('freq = %r' % freq)
        #self.end(self.count + 1)

    #def make_substep_progiters():

    #def iter_without_rate(self):
    #    if self.mark is None:
    #        # Continuation of hacking
    #        self.mark, self.end = log_progress(*self.args, **self.kwargs)
    #    mark = self.mark
    #    # Wrap the for loop with a generator
    #    self.count = -1
    #    for self.count, item in enumerate(self.iterable):
    #        mark(self.count)
    #        yield item
    #    self.end(self.count + 1)

    #def mark_current(self):
    #    self.mark(self.count)

    def _get_timethresh_heuristics(self):
        """
        resonably decent hueristics for how much time to wait before
        updating progress.
        """
        if self.nTotal > 1E5:
            time_thresh = 2.5
        elif self.nTotal > 1E4:
            time_thresh = 2.0
        elif self.nTotal > 1E3:
            time_thresh = 1.0
        else:
            time_thresh = 0.5
        return time_thresh


progiter = ProgressIter


class ProgIter(ProgressIter):
    """ Thin wrapper with better arg positions """
    def __init__(self, iterable, lbl='Prog', adjust=True, freq=1, bs=True,
                 **kwargs):
        import utool as ut
        super(ut.ProgIter, self).__init__(iterable, lbl=lbl, adjust=adjust,
                                          freq=freq, bs=bs, **kwargs)


def progress_str(max_val, lbl='Progress: ', repl=False, approx=False, backspace=PROGGRESS_BACKSPACE):
    r""" makes format string that prints progress: %Xd/MAX_VAL with backspaces

    NOTE: \r can be used instead of backspaces. This function is not very
    relevant because of that.

    """
    # string that displays max value
    max_str = six.text_type(max_val)
    if approx:
        # denote approximate maximum
        max_str = '~' + max_str
    dnumstr = six.text_type(len(max_str))
    # string that displays current progress
    cur_str = '%' + dnumstr + 'd'
    # If user passed in the label
    if repl:
        _fmt_str = lbl.replace('<cur_str>', cur_str).replace('<max_str>', max_str)
    else:
        _fmt_str = lbl + cur_str + '/' + max_str
    if backspace:
        # put backspace characters into the progress string
        # (looks nice on normal terminals)
        #nBackspaces = len(_fmt_str) - len(dnumstr) + len(max_str)
        #backspaces = '\b' * nBackspaces
        #fmt_str = backspaces + _fmt_str
        # FIXME: USE CARAGE RETURN INSTEAD OF BACKSPACES
        fmt_str = '\r' + _fmt_str
    else:
        # FIXME: USE CARAGE RETURN INSTEAD OF BACKSPACES
        # this looks better on terminals without backspaces
        fmt_str = _fmt_str + '\n'
    return fmt_str


def log_progress(lbl='Progress: ', nTotal=0, flushfreq=4, startafter=-1,
                 start=True, repl=False, approx=False, disable=False,
                 writefreq=1, with_time=False, backspace=True,
                 pad_stdout=False, wfreq=None, ffreq=None, freq=None, total=None,
                 num=None, with_totaltime=None):
    """
    DEPRICATE
    FIXME: depricate for ProgressIter.
    still used in util_dev
    """
    global AGGROFLUSH
    # Alias kwargs with simpler names
    if num is not None:
        nTotal = num
    if total is not None:
        nTotal = total
    if wfreq is not None:
        writefreq = wfreq
    if ffreq is not None:
        flushfreq = ffreq
    if freq is not None:
        writefreq = flushfreq = freq
    if with_totaltime is not None:
        with_time = with_totaltime
    # flush frequency must be a multiple of write frequency
    flushfreq = max(int(round(flushfreq / writefreq)), 1) * writefreq
    if nTotal < startafter or disable:
        # Do not mark progress if only executing a small number of tasks
        def mark_progress(*args):
            pass
        def end_progress(*args):
            pass
        return mark_progress, end_progress
    else:
        write_fn = PROGRESS_WRITE
        flush_fn = PROGRESS_FLUSH
        # build format string for displaying progress
        fmt_str = progress_str(nTotal, lbl=lbl, repl=repl, approx=approx,
                               backspace=backspace)
        if AGGROFLUSH:
            # Progress function which automatically flushes
            def mark_progress(count, flush_fn=flush_fn):
                count_ = count + 1
                write_fn(fmt_str % (count_))
                flush_fn()
        else:
            # Progress function flushes every <flushfreq> times
            def mark_progress(count, fmt_str=fmt_str, flushfreq=flushfreq,
                              writefreq=writefreq, write_fn=write_fn,
                              flush_fn=flush_fn):
                count_ = count + 1
                if count_ % writefreq == 0:
                    write_fn(fmt_str % count_)
                    if count_ % flushfreq == 0:
                        flush_fn()

        if pad_stdout:
            write_fn('\n')
            write_fn('\n')
            flush_fn()

        if with_time:
            tt = util_time.tic(lbl)

        def end_progress(count_=nTotal, write_fn=write_fn, flush_fn=flush_fn):
            write_fn(fmt_str % (count_))
            write_fn('\n')
            flush_fn()
            if with_time:
                util_time.toc(tt)
            if pad_stdout:
                write_fn('\n\n')
                flush_fn()
        #mark_progress(0)
        if start:
            mark_progress(-1)
        return mark_progress, end_progress


def simple_progres_func(verbosity, msg, progchar='.'):
    def mark_progress0(*args):
        pass

    def mark_progress1(*args):
        PROGRESS_WRITE(progchar)

    def mark_progress2(*args):
        print(msg % args)

    if verbosity == 0:
        mark_progress = mark_progress0
    elif verbosity == 1:
        mark_progress = mark_progress1
    elif verbosity == 2:
        mark_progress = mark_progress2
    return mark_progress


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_progress; utool.doctest_funcs(utool.util_progress, allexamples=True)"
        python -c "import utool, utool.util_progress; utool.doctest_funcs(utool.util_progress)"
        python -m utool.util_progress
        python -m utool.util_progress --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
