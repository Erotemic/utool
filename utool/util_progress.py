"""
# Need to invoke this way because of relative imports


Examples:
    python -c "import utool, doctest; print(doctest.testmod(utool.util_progress))"
    python -m doctest -v ~/code/utool/utool/util_progress.py
    python -c "import utool, doctest; doctest.testmod(utool.util_progress, verbose=True)"
    python -c "import utool, doctest; doctest.testmod(utool.util_progress)" -v
    python -c "import utool, doctest; doctest.testmod(utool.util_progress)"
    python -c "import utool, doctest; help(doctest.testmod)"
"""
from __future__ import absolute_import, division, print_function
import time
import sys
from utool import util_logging
from utool.util_inject import inject
from utool.util_arg import QUIET, SILENT
from utool import util_time
#get_argflag,
#, VERBOSE
print, print_, printDBG, rrr, profile = inject(__name__, '[progress]')


#QUIET = get_argflag('--quiet')
#VERBOSE = get_argflag('--verbose')
VALID_PROGRESS_TYPES = ['none', 'dots', 'fmtstr', 'simple']
AGGROFLUSH = '--aggroflush' in sys.argv
PROGGRESS_BACKSPACE = ('--screen' not in sys.argv and '--progress-backspace' not in sys.argv)


#PROGRESS_WRITE = sys.stdout.write
#PROGRESS_FLUSH = sys.stdout.flush

PROGRESS_WRITE = util_logging.__UTOOL_WRITE__
PROGRESS_FLUSH = util_logging.__UTOOL_FLUSH__


def test_progress():
    """
    Example:
        >>> # ENABLE_DOCTEST
        >>> test_progress()
    """
    import utool as ut
    import time
    ut.rrrr()

    print('_________________')

    numiter = 1000
    rate = .01

    with ut.Timer():
        for x in ut.ProgressIter(range(0, numiter), freq=4):
            time.sleep(rate)

    print('_________________')
    print('No frequncy run:')

    with ut.Timer():
        for x in range(0, numiter):
            time.sleep(rate)

    print('_________________')

    numiter = 100000
    rate = .000008

    with ut.Timer():
        for x in ut.ProgressIter(range(0, numiter), freq=4):
            time.sleep(rate)

    print('_________________')

    with ut.Timer():
        for x in ut.ProgressIter(range(0, numiter), freq=100):
            time.sleep(rate)

    print('_________________')
    print('No frequncy run:')

    with ut.Timer():
        for x in range(0, numiter):
            time.sleep(rate)

    print('_________________')


class ProgressIter(object):
    """
    Wraps a for loop with progress reporting

    lbl='Progress: ', nTotal=0, flushfreq=4, startafter=-1, start=True,
    repl=False, approx=False, disable=False, writefreq=1, with_time=False,
    backspace=True, separate=False, wfreq=None, ffreq=None, freq=None,
    total=None, num=None, with_totaltime=None

    Args:
        iterable (): iterable normally passed to for loop
        lbl (str):  progress label
        nTotal (int):
        flushfreq (int):
        startafter (int):
        start (bool):
        repl (bool):
        approx (bool):
        disable (bool):
        writefreq (int):
        with_totaltime (bool):
        backspace (bool):
        separate (bool):
        wfreq (None): alias for write_freq
        ffreq (None): alias for flush_freq
        total (None): alias for nTotal
        num (None):   alias for nTotal

    Example:
        >>> import utool
        >>> from six.moves import range
        >>> results1 = [x for x in utool.ProgressIter(range(10000), wfreq=10)]
        >>> results4 = [x for x in utool.ProgressIter(range(10000), wfreq=1)]
        >>> results2 = [x for x in range(10000)]
        >>> results3 = [x for x in utool.progiter((y + 1 for y in range(1000001)), nTotal=1000001, wfreq=1000, backspace=True)]
        >>> assert results1 == results2

    """
    def new_init(self, iterable=None, lbl='ProgIter', nTotal=None, freq=4,
                 newlines=False):
        pass

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        if len(args) < 2 and 'nTotal' not in kwargs:
            try:
                nTotal = len(iterable)
                kwargs['nTotal'] = nTotal
            except Exception:
                pass
        self.time_thresh = kwargs.pop('time_thresh', None)
        self.use_rate  = kwargs.pop('use_rate', True)
        self.lbl       = kwargs.get('lbl', 'lbl')
        self.nTotal    = kwargs.get('nTotal', 0)
        self.backspace = kwargs.get('backspace', True)
        self.freq      = kwargs.get('freq', 4)
        self.mark, self.end = log_progress(*args, **kwargs)
        self.count = -1

    def __call__(self, iterable):
        self.iterable = iterable
        return self

    def __iter__(self):
        if self.use_rate:
            return self.iter_rate()
        else:
            return self.iter_without_rate()

    def iter_without_rate(self):
        mark = self.mark
        # Wrap the for loop with a generator
        self.count = -1
        for self.count, item in enumerate(self.iterable):
            mark(self.count)
            yield item
        self.end(self.count + 1)

    def iter_rate(self):
        # TODO Incorporate this better
        import utool as ut
        starttime = time.time()
        last_time = starttime
        #mark = self.mark
        # Wrap the for loop with a generator
        self.count = -1
        freq = self.freq
        #cumrate = 1E-9
        between_count = 0
        last_count = -1
        #self.nTotal = len(self.iterable)
        nTotal = self.nTotal
        fmt_msg = ''.join(('\r', self.lbl,
                           ' %4d/', str(nTotal),
                           '...  rate=%4.1f iters per second.',
                           ' est_min_left=%4.2f              '))

        if not self.backspace:
            fmt_msg += '\n'

        # how long iterations should be before a flush
        if self.time_thresh is None:
            if self.nTotal > 1E5:
                time_thresh = 4.0
            elif self.nTotal > 1E4:
                time_thresh = 2.0
            elif self.nTotal > 1E3:
                time_thresh = 1.0
            else:
                time_thresh = 0.5
        else:
            time_thresh = self.time_thresh
        #time_thresh = 0.5
        max_between_time = -1.0
        max_between_count = -1.0  # why is this different? # becuase frequency varies

        with ut.Timer(self.lbl):
            import six
            # yeild first element
            enumiter = enumerate(self.iterable)
            yield six.next(enumiter)[1]

            for self.count, item in enumiter:
                #mark(self.count)
                yield item
                if (self.count) % freq == 0:
                    between_count = self.count - last_count
                    now_time = time.time()
                    between_time = (now_time - last_time)
                    # Adjust frequency if printing too quickly
                    # so progress doesnt slow down actual function
                    # TODO: better adjust algorithm
                    if between_time < time_thresh:
                        #print('')
                        #print('[prog] Adusting frequency from: %r' % freq)
                        #print('between_count = %r' % between_count)
                        #print('between_time = %r' % between_time)
                        # There has to be a standard way to do this.
                        # Refer to: https://github.com/verigak/progress/blob/master/progress/__init__.py
                        max_between_time = max(max(max_between_time, between_time), 1E-9)
                        max_between_count = max(max_between_count, between_count)
                        #freq = max(int(1.3 * between_count * time_thresh / between_time), 1)
                        freq = max(int(1.3 * max_between_count * time_thresh / max_between_time), 1)
                        #freq = max(int((between_count * between_time) / time_thresh), 1)
                        #freq = max(int((between_count) / time_thresh), 1)
                        #freq = max(int((between_time) / time_thresh), 1)
                        #freq = max(int(time_thresh / between_count), 1)
                        #print('[prog] Adusting frequency to: %r' % freq)
                        #print('')
                    iters_per_second = between_count / (float(between_time) + 1E-9)
                    #cumrate += between_time
                    #rate = (self.count + 1.0) / float(cumrate)
                    iters_left = nTotal - self.count
                    est_seconds_left = iters_left / (iters_per_second + 1E-9)
                    est_min_left = est_seconds_left / 60.0
                    msg = fmt_msg % (self.count, iters_per_second, est_min_left)
                    #if False and __debug__:
                    #    print('<!!!!!!!!!!!!!>')
                    #    print('iters_left = %r' % iters_left)
                    #    print('between_time = %r' % between_time)
                    #    print('between_count = %r' % between_count)
                    #    print('est_seconds_left = %r' % est_seconds_left)
                    #    print('iters_per_second = %r' % iters_per_second)
                    #    print('</!!!!!!!!!!!!!>')
                    PROGRESS_WRITE(msg)
                    PROGRESS_FLUSH()
                    last_count = self.count
                    last_time = now_time
            #print('freq = %r' % freq)
            PROGRESS_WRITE('\n')
            PROGRESS_FLUSH()
        #self.end(self.count + 1)

    def mark_current(self):
        self.mark(self.count)


progiter = ProgressIter


def progress_str(max_val, lbl='Progress: ', repl=False, approx=False, backspace=PROGGRESS_BACKSPACE):
    r""" makes format string that prints progress: %Xd/MAX_VAL with backspaces

    NOTE: \r can be used instead of backspaces. This function is not very
    relevant because of that.

    """
    # string that displays max value
    max_str = str(max_val)
    if approx:
        # denote approximate maximum
        max_str = '~' + max_str
    dnumstr = str(len(max_str))
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
                 separate=False, wfreq=None, ffreq=None, freq=None, total=None,
                 num=None, with_totaltime=None):
    """
    Returns two functions (mark_progress, end_progress) which will handle
    logging progress in a for loop.

    flush frequency must be a multiple of write frequency

    Args:
        lbl (str):  progress label
        nTotal (int):
        flushfreq (int):
        startafter (int):
        start (bool):
        repl (bool):
        approx (bool):
        disable (bool):
        writefreq (int):
        with_totaltime (bool):
        backspace (bool):
        separate (bool):
        wfreq (None): alias for write_freq
        ffreq (None): alias for flush_freq
        freq (None):  alias for flush_freq and write_freq (prefered)
        total (None): alias for nTotal
        num (None):   alias for nTotal

    Example:
        >>> import utool, time
        >>> from six.moves import range
        >>> # Define a dummy task
        >>> spam = 42.0
        >>> nTotal = 1000
        >>> iter_ = (num for num in range(0, nTotal * 2, 2))
        >>> # Create progress functions
        ... mark_, end_ = utool.log_progress('prog ', nTotal, flushfreq=17)
        \b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bprog    0/1000
        >>> for count, item in enumerate(iter_):  #doctest: +ELLIPSIS
        ...     # Call with enumerate to keep track of a count variable
        ...     time.sleep(.001)
        ...     spam += item + count
        ...     mark_(count)
        \b...prog 1000/1000
        >>> # Mark completion
        >>> end_()
        <BLANKLINE>
    """
    # utool.auto_docstr('utool.util_progress', 'log_progress')
    # python -c "import utool; utool.print_auto_docstr('utool.util_progress', 'log_progress')"
    #
    # In reference to above docstr:
    #    I don't completely understand why some of the >>> and ... had to be where
    #    they are, but doctest gets very angry if its not in this format

    # TODO: Option to display rate of progress
    # TODO: Option to display eta
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

        if separate:
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
            if separate:
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


def prog_func(*args, **kwargs):
    return progress_func(*args, **kwargs)


# TODO: Return start_prog, make_prog, end_prog
def progress_func(max_val=0, lbl='Progress: ', mark_after=-1,
                  flush_after=4, spacing=0, line_len=80,
                  progress_type='fmtstr', mark_start=False, repl=False,
                  approx=False, override_quiet=False):
    """ DEPRICATE

    Returns:
        a function that marks progress taking the iteration count as a
        parameter. Prints if max_val > mark_at. Prints dots if max_val not
        specified or simple=True
    """
    write_fn = PROGRESS_WRITE
    #write_fn = print_
    #print('STARTING PROGRESS: VERBOSE=%r QUIET=%r' % (VERBOSE, QUIET))

    # Tell the user we are about to make progress
    if SILENT or (QUIET and not override_quiet) or (progress_type in ['simple', 'fmtstr'] and max_val < mark_after):
        return lambda count: None, lambda: None
    # none: nothing
    if progress_type == 'none':
        mark_progress =  lambda count: None
    # simple: one dot per progress. no flush.
    if progress_type == 'simple':
        mark_progress = lambda count: write_fn('.')
    # dots: spaced dots
    if progress_type == 'dots':
        indent_ = '    '
        write_fn(indent_)

        if spacing > 0:
            # With spacing
            newline_len = spacing * line_len // spacing

            def mark_progress_sdot(count):
                write_fn('.')
                count_ = count + 1
                if (count_) % newline_len == 0:
                    write_fn('\n' + indent_)
                    PROGRESS_FLUSH()
                elif (count_) % spacing == 0:
                    write_fn(' ')
                    PROGRESS_FLUSH()
                elif (count_) % flush_after == 0:
                    PROGRESS_FLUSH()
            mark_progress = mark_progress_sdot
        else:
            # No spacing
            newline_len = line_len

            def mark_progress_dot(count):
                write_fn('.')
                count_ = count + 1
                if (count_) % newline_len == 0:
                    write_fn('\n' + indent_)
                    PROGRESS_FLUSH()
                elif (count_) % flush_after == 0:
                    PROGRESS_FLUSH()
            mark_progress = mark_progress_dot
    # fmtstr: formated string progress
    if progress_type == 'fmtstr':
        fmt_str = progress_str(max_val, lbl=lbl, repl=repl, approx=approx)

        def mark_progress_fmtstr(count):
            count_ = count + 1
            write_fn(fmt_str % (count_))
            if (count_) % flush_after == 0:
                PROGRESS_FLUSH()
        mark_progress = mark_progress_fmtstr
    # FIXME idk why argparse2.ARGS_ is none here.
    if '--aggroflush' in sys.argv:
        def mark_progress_agressive(count):
            mark_progress(count)
            PROGRESS_FLUSH()
        return mark_progress_agressive

    def end_progress():
        write_fn('\n')
        PROGRESS_FLUSH()
    #mark_progress(0)
    if mark_start:
        mark_progress(-1)
    return mark_progress, end_progress
    raise Exception('unkown progress type = %r' % progress_type)


if __name__ == '__main__':
    """
    CommandLine:
        python utool/util_progress.py --test-test-progress
    """
    from utool.util_tests import doctest_funcs
    doctest_funcs()
