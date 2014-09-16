"""
# Need to invoke this way because of relative imports
python -c "import utool, doctest; print(doctest.testmod(utool.util_progress))"

python -m doctest -v ~/code/utool/utool/util_progress.py
python -c "import utool, doctest; doctest.testmod(utool.util_progress, verbose=True)"
python -c "import utool, doctest; doctest.testmod(utool.util_progress)" -v
python -c "import utool, doctest; doctest.testmod(utool.util_progress)"
python -c "import utool, doctest; help(doctest.testmod)"
"""
from __future__ import absolute_import, division, print_function
import sys
from .util_inject import inject
from .util_arg import QUIET
from . import util_time
#get_flag,
#, VERBOSE
print, print_, printDBG, rrr, profile = inject(__name__, '[progress]')


#QUIET = get_flag('--quiet')
#VERBOSE = get_flag('--verbose')
VALID_PROGRESS_TYPES = ['none', 'dots', 'fmtstr', 'simple']
AGGROFLUSH = '--aggroflush' in sys.argv
PROGGRESS_BACKSPACE = '--screen' not in sys.argv


def log_progress(lbl='Progress: ', nTotal=0, flushfreq=4, startafter=-1,
                 start=True, repl=False, approx=False, disable=False,
                 writefreq=1, with_totaltime=False):
    """
    Returns two functions (mark_progress, end_progress) which will handle
    logging progress in a for loop.
    </CYTH>

    # Example / Doctest
    # I don't completely understand why some of the >>> and ... had to be where
    # they are, but doctest gets very angry if its not in this format
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
    global AGGROFLUSH
    if nTotal < startafter or disable:
        # Do not mark progress if only executing a small number of tasks
        def mark_progress(*args):
            pass
        def end_progress(*args):
            pass
        return mark_progress, end_progress
    else:
        write_fn = sys.stdout.write
        flush_fn = sys.stdout.flush
        # build format string for displaying progress
        fmt_str = progress_str(nTotal, lbl=lbl, repl=repl, approx=approx)
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

        if with_totaltime:
            tt = util_time.tic(lbl)

        def end_progress(write_fn=write_fn, flush_fn=flush_fn):
            write_fn(fmt_str % (nTotal))
            write_fn('\n')
            flush_fn()
            if with_totaltime:
                util_time.toc(tt)
        #mark_progress(0)
        if start:
            mark_progress(-1)
        return mark_progress, end_progress


def simple_progres_func(verbosity, msg, progchar='.'):
    """ </CYTH> """
    def mark_progress0(*args):
        pass

    def mark_progress1(*args):
        sys.stdout.write(progchar)

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
    """ </CYTH> """
    return progress_func(*args, **kwargs)


# TODO: Return start_prog, make_prog, end_prog
def progress_func(max_val=0, lbl='Progress: ', mark_after=-1,
                  flush_after=4, spacing=0, line_len=80,
                  progress_type='fmtstr', mark_start=False, repl=False,
                  approx=False, override_quiet=False):
    """Returns a function that marks progress taking the iteration count as a
    parameter. Prints if max_val > mark_at. Prints dots if max_val not
    specified or simple=True
    </CYTH>
    """
    write_fn = sys.stdout.write
    #write_fn = print_
    #print('STARTING PROGRESS: VERBOSE=%r QUIET=%r' % (VERBOSE, QUIET))

    # Tell the user we are about to make progress
    if (QUIET and not override_quiet) or (progress_type in ['simple', 'fmtstr'] and max_val < mark_after):
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
                    sys.stdout.flush()
                elif (count_) % spacing == 0:
                    write_fn(' ')
                    sys.stdout.flush()
                elif (count_) % flush_after == 0:
                    sys.stdout.flush()
            mark_progress = mark_progress_sdot
        else:
            # No spacing
            newline_len = line_len

            def mark_progress_dot(count):
                write_fn('.')
                count_ = count + 1
                if (count_) % newline_len == 0:
                    write_fn('\n' + indent_)
                    sys.stdout.flush()
                elif (count_) % flush_after == 0:
                    sys.stdout.flush()
            mark_progress = mark_progress_dot
    # fmtstr: formated string progress
    if progress_type == 'fmtstr':
        fmt_str = progress_str(max_val, lbl=lbl, repl=repl, approx=approx)

        def mark_progress_fmtstr(count):
            count_ = count + 1
            write_fn(fmt_str % (count_))
            if (count_) % flush_after == 0:
                sys.stdout.flush()
        mark_progress = mark_progress_fmtstr
    # FIXME idk why argparse2.ARGS_ is none here.
    if '--aggroflush' in sys.argv:
        def mark_progress_agressive(count):
            mark_progress(count)
            sys.stdout.flush()
        return mark_progress_agressive

    def end_progress():
        write_fn('\n')
        sys.stdout.flush()
    #mark_progress(0)
    if mark_start:
        mark_progress(-1)
    return mark_progress, end_progress
    raise Exception('unkown progress type = %r' % progress_type)


def progress_str(max_val, lbl='Progress: ', repl=False, approx=False):
    """ makes format string that prints progress: %Xd/MAX_VAL with backspaces
    </CYTH>
    """
    max_str = str(max_val)
    if approx:
        # denote approximate maximum
        max_str = '~' + max_str
    dnumstr = str(len(max_str))
    cur_str = '%' + dnumstr + 'd'
    if repl:
        fmt_str_ = lbl.replace('<cur_str>', cur_str).replace('<max_str>', max_str)
    else:
        fmt_str_ = lbl + cur_str + '/' + max_str
    if PROGGRESS_BACKSPACE:
        fmt_str = '\b' * (len(fmt_str_) - len(dnumstr) + len(max_str)) + fmt_str_
    else:
        fmt_str = fmt_str_ + '\n'
    return fmt_str
