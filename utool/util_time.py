"""
TODO: This file seems to care about timezone

TODO: Use UTC/GMT time here for EVERYTHING

References:
    http://www.timeanddate.com/time/aboututc.html

"""
from __future__ import absolute_import, division, print_function
import sys
import six
import time
import datetime
from utool import util_inject
print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[time]')


# --- Timing ---
def tic(msg=None):
    return (msg, time.time())


def toc(tt, return_msg=False, write_msg=True):
    (msg, start_time) = tt
    ellapsed = (time.time() - start_time)
    if (not return_msg) and write_msg and msg is not None:
        sys.stdout.write('...toc(%.4fs, ' % ellapsed + '"' + str(msg) + '"' + ')\n')
    if return_msg:
        return msg
    else:
        return ellapsed


def get_printable_timestamp():
    return get_timestamp('printable')


def get_timestamp(format_='filename', use_second=False, delta_seconds=None):
    """
    get_timestamp

    Args:
        format_ (str):
        use_second (bool):
        delta_seconds (None):

    Returns:
        str: stamp

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> format_ = 'printable'
        >>> use_second = False
        >>> delta_seconds = None
        >>> stamp = get_timestamp(format_, use_second, delta_seconds)
        >>> print(stamp)
        >>> assert len(stamp) == len('15:43:04 2015/02/24')
    """
    now = datetime.datetime.now()
    # TODO: time.timezone
    #now = datetime.datetime.utcnow()
    if delta_seconds is not None:
        now += datetime.timedelta(seconds=delta_seconds)
    if format_ == 'tag':
        time_tup = (now.year - 2000, now.month, now.day)
        stamp = '%02d%02d%02d' % time_tup
    elif format_ == 'printable':
        time_tup = (now.hour, now.minute, now.second, now.year, now.month, now.day)
        time_format = '%02d:%02d:%02d %02d/%02d/%02d'
        stamp = time_format % time_tup
    else:
        if use_second:
            time_tup = (now.year, now.month, now.day, now.hour, now.minute, now.second)
            time_formats = {
                'filename': 'ymd_hms-%04d-%02d-%02d_%02d-%02d-%02d',
                'comment': '# (yyyy-mm-dd hh:mm:ss) %04d-%02d-%02d %02d:%02d:%02d'}
        else:
            time_tup = (now.year, now.month, now.day, now.hour, now.minute)
            time_formats = {
                'filename': 'ymd_hm-%04d-%02d-%02d_%02d-%02d',
                'comment': '# (yyyy-mm-dd hh:mm) %04d-%02d-%02d %02d:%02d'}
        stamp = time_formats[format_] % time_tup
    return stamp


def get_datestamp(explicit=True):
    now = datetime.datetime.now()
    stamp = '%04d-%02d-%02d' % (now.year, now.month, now.day)
    if explicit:
        return 'ymd-' + stamp + time.timezone[0]
    else:
        return stamp


# alias
timestamp = get_timestamp


class Timer(object):
    """
    Timer with-statment context object.

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool
        >>> with utool.Timer('Timer test!'):
        >>>     prime = utool.get_nth_prime(400)
    """
    def __init__(self, msg='', verbose=True, newline=True):
        self.msg = msg
        self.verbose = verbose
        self.newline = newline
        self.tstart = -1
        self.ellapsed = -1
        #self.tic()

    def tic(self):
        if self.verbose:
            sys.stdout.flush()
            print_('\ntic(%r)' % self.msg)
            if self.newline:
                print_('\n')
            sys.stdout.flush()
        self.tstart = time.time()

    def toc(self):
        ellapsed = (time.time() - self.tstart)
        if self.verbose:
            print_('...toc(%r)=%.4fs\n' % (self.msg, ellapsed))
            sys.stdout.flush()
        return ellapsed

    def __enter__(self):
        #if self.msg is not None:
        #    sys.stdout.write('---tic---' + self.msg + '  \n')
        self.tic()
        return self

    def __exit__(self, type_, value, trace):
        self.ellapsed = self.toc()
        if trace is not None:
            #print('[util_time] Error in context manager!: ' + str(value))
            pass
            return False  # return a falsey value on error
        #return self.ellapsed


def exiftime_to_unixtime(datetime_str, timestamp_format=1, strict=False):
    """
    converts a datetime string to posixtime (unixtime)

    Args:
        datetime_str     (str):
        timestamp_format (int):

    Returns:
        int: unixtime seconds from 1970

    CommandLine:
        python -m utool.util_time --test-exiftime_to_unixtime

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> datetime_str = '0000:00:00 00:00:00'
        >>> timestamp_format = 1
        >>> result = exiftime_to_unixtime(datetime_str, timestamp_format)
        >>> print(result)
        -1

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> datetime_str = '2015:04:01 00:00:00'
        >>> timestamp_format = 1
        >>> result = exiftime_to_unixtime(datetime_str, timestamp_format)
        >>> print(result)
        1427860800.0
    """
    try:
        # Normal format, or non-standard year first data
        if timestamp_format == 2:
            timefmt = '%m/%d/%Y %H:%M:%S'
        elif timestamp_format == 1:
            timefmt = '%Y:%m:%d %H:%M:%S'
        else:
            assert isinstance(timestamp_format, six.string_types)
            timefmt = timestamp_format
            #raise AssertionError('unknown timestamp_format=%r' % (timestamp_format,))
        if len(datetime_str) > 19:
            datetime_str_ = datetime_str[:19].strip(';').strip()
        else:
            datetime_str_ = datetime_str
        dt = datetime.datetime.strptime(datetime_str_, timefmt)
        return time.mktime(dt.timetuple())
    except TypeError:
        #if datetime_str is None:
            #return -1
        return -1
    except ValueError as ex:
        #from utool.util_arg import STRICT
        if isinstance(datetime_str, six.string_types):
            if datetime_str_.find('No EXIF Data') == 0:
                return -1
            if datetime_str_.find('Invalid') == 0:
                return -1
            if datetime_str_ == '0000:00:00 00:00:00':
                return -1
        print('<!!! ValueError !!!>')
        print('[util_time] Caught Error: ' + repr(ex))
        print('[util_time] type(datetime_str)  = %r' % type(datetime_str))
        print('[util_time] repr(datetime_str)  = %r' % datetime_str)
        print('[util_time]     (datetime_str)  = %s' % datetime_str)
        print('[util_time]  len(datetime_str)  = %d' % len(datetime_str))
        print('[util_time] repr(datetime_str_) = %r' % datetime_str_)
        print('[util_time]  len(datetime_str_) = %d' % len(datetime_str_))
        print('</!!! ValueError !!!>')
        if strict:
            raise
        else:
            print('Supressed ValueError')
            return -1


def unixtime_to_datetime(unixtime, timefmt='%Y/%m/%d %H:%M:%S'):
    if unixtime == -1:
        return 'NA'
    if unixtime is None:
        return None
    return datetime.datetime.fromtimestamp(unixtime).strftime(timefmt)


def unixtime_to_timedelta(unixtime_diff):
    """ alias for get_unix_timedelta """
    return get_unix_timedelta(unixtime_diff)


def get_unix_timedelta(unixtime_diff):
    timedelta = datetime.timedelta(seconds=abs(unixtime_diff))
    return timedelta


def get_unix_timedelta_str(unixtime_diff):
    """
    Args:
        unixtime_diff (int): number of seconds

    Returns:
        timestr (str): formated time string

    Args:
        unixtime_diff (int):

    Returns:
        str: timestr

    CommandLine:
        python -m utool.util_time --test-get_unix_timedelta_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> unixtime_diff = 0
        >>> timestr = get_unix_timedelta_str(unixtime_diff)
        >>> timestr_list = [get_unix_timedelta_str(_) for _ in [-9001, -1, 0, 1, 9001]]
        >>> result = str(timestr_list)
        >>> print(result)
        ['2 hours 30 minutes 1 seconds', '1 seconds', '0 seconds', '1 seconds', '2 hours 30 minutes 1 seconds']
    """
    timedelta = get_unix_timedelta(unixtime_diff)
    timestr = get_timedelta_str(timedelta)
    return timestr


def get_timedelta_str(timedelta):
    """
    get_timedelta_str

    Returns:
        str: timedelta_str, formated time string

    References:
        http://stackoverflow.com/questions/8906926/formatting-python-timedelta-objects

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> timedelta = get_unix_timedelta(10)
        >>> timedelta_str = get_timedelta_str(timedelta)
        >>> result = (timedelta_str)
        >>> print(result)
        10 seconds
    """
    if timedelta == datetime.timedelta(0):
        return '0 seconds'
    days = timedelta.days
    hours, rem = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    fmtstr_list = []
    fmtdict = {}
    if abs(days) > 0:
        fmtstr_list.append('{days} days')
        fmtdict['days'] = days
    if len(fmtstr_list) > 0 or abs(hours) > 0:
        fmtstr_list.append('{hours} hours')
        fmtdict['hours'] = hours
    if len(fmtstr_list) > 0 or abs(minutes) > 0:
        fmtstr_list.append('{minutes} minutes')
        fmtdict['minutes'] = minutes
    if len(fmtstr_list) > 0 or abs(seconds) > 0:
        fmtstr_list.append('{seconds} seconds')
        fmtdict['seconds'] = seconds
    fmtstr = ' '.join(fmtstr_list)
    timedelta_str = fmtstr.format(**fmtdict)
    return timedelta_str


def get_posix_timedelta_str(posixtime):
    """
    get_timedelta_str

    Returns:
        str: timedelta_str, formated time string

    CommandLine:
        python -m utool.util_time --test-get_posix_timedelta_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> posixtime_list = [-13, 10.2, 10.2 ** 2, 10.2 ** 3, 10.2 ** 4, 10.2 ** 5, 10.2 ** 8]
        >>> posixtime = posixtime_list[1]
        >>> timedelta_str = [get_posix_timedelta_str(posixtime) for posixtime in posixtime_list]
        >>> result = (timedelta_str)
        >>> print(result)
        ['-00:00:13', '00:00:10.20', '00:01:44.04', '00:17:41.21', '03:00:24.32', '1 days 06:40:08.08', '193 weeks 5 days 02:05:38.10']

    Timeit::
        import datetime
        # Seems like like timedelta is just faster. must be because it is builtin
        %timeit get_posix_timedelta_str(posixtime)
        %timeit str(datetime.timedelta(seconds=posixtime))

    """
    sign, posixtime_ = (1, posixtime) if posixtime >= 0 else (-1, -posixtime)
    seconds_, subseconds = divmod(posixtime_, 1)
    minutes_, seconds    = divmod(int(seconds_), 60)
    hours_, minutes      = divmod(minutes_, 60)
    days_, hours         = divmod(hours_, 24)
    weeks_, days         = divmod(days_, 7)
    timedelta_str = ':'.join(['%02d' % _ for _ in (hours, minutes, seconds)])
    if subseconds > 0:
        timedelta_str += ('%.2f' % (subseconds,))[1:]
    if days_ > 0:
        timedelta_str = '%d days ' % (days,) + timedelta_str
    if weeks_ > 0:
        timedelta_str = '%d weeks ' % (weeks_,) + timedelta_str
    if sign == -1:
        timedelta_str = '-' + timedelta_str
    return timedelta_str


#def get_simple_posix_timedelta_str(posixtime):
#    """
#    get_timedelta_str

#    Returns:
#        str: timedelta_str, formated time string

#    CommandLine:
#        python -m utool.util_time --test-get_posix_timedelta_str

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from utool.util_time import *  # NOQA
#        >>> posixtime_list = [13, 10.2, 10.2 ** 2, 10.2 ** 3, 10.2 ** 4, 10.2 ** 5, 10.2 ** 8]
#        >>> posixtime = posixtime_list[1]
#        >>> timedelta_str = [get_simple_posix_timedelta_str(posixtime) for posixtime in posixtime_list]
#        >>> result = (timedelta_str)
#        >>> print(result)

#    Timeit::
#        import datetime
#        posixtime = 10.2 ** 8
#        %timeit get_simple_posix_timedelta_str(posixtime)
#        %timeit str(datetime.timedelta(seconds=posixtime))

#    """
#    seconds_ = int(posixtime)
#    minutes_, seconds    = divmod(seconds_, 60)
#    hours_, minutes      = divmod(minutes_, 60)
#    days_, hours         = divmod(hours_, 24)
#    weeks_, days         = divmod(days_, 7)
#    timedelta_str = ':'.join(['%02d' % _ for _ in (hours, minutes, seconds)])
#    #if days_ > 0:
#    #    timedelta_str = '%d days ' % (days,) + timedelta_str
#    #if weeks_ > 0:
#    #    timedelta_str = '%d weeks ' % (weeks_,) + timedelta_str
#    return timedelta_str


def get_month():
    return datetime.datetime.now().month


def get_day():
    return datetime.datetime.now().day


def get_year():
    return datetime.datetime.now().year


def get_timestats_str(unixtime_list, newlines=False):
    r"""
    Args:
        unixtime_list (list):
        newlines (bool):

    Returns:
        str: timestat_str

    CommandLine:
        python -m utool.util_time --test-get_timestats_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> import utool as ut
        >>> # build test data
        >>> # TODO: FIXME ME FOR TIMEZONE EST vs GMT
        >>> unixtime_list = [0 + 60*60*5 , 10+ 60*60*5, 100+ 60*60*5, 1000+ 60*60*5]
        >>> newlines = True
        >>> # execute function
        >>> timestat_str = get_timestats_str(unixtime_list, newlines)
        >>> # verify results
        >>> result = ut.align(str(timestat_str), ':')
        >>> print(result)
        {
            'std'  : '0:06:59',
            'max'  : '1970/01/01 00:16:40',
            'range': '0:16:40',
            'mean' : '1970/01/01 00:04:37',
            'min'  : '1970/01/01 00:00:00',
        }

    """
    import utool as ut
    datetime_stats = get_timestats_dict(unixtime_list)
    timestat_str = ut.dict_str(datetime_stats, newlines=newlines)
    return timestat_str


def get_timestats_dict(unixtime_list):
    import utool as ut
    unixtime_stats = ut.get_stats(unixtime_list)
    datetime_stats = {}
    for key in ['max', 'min', 'mean']:
        try:
            datetime_stats[key] = ut.unixtime_to_datetime(unixtime_stats[key])
        except KeyError:
            pass
    for key in ['std']:
        try:
            datetime_stats[key] = str(ut.unixtime_to_timedelta(int(round(unixtime_stats[key]))))
        except KeyError:
            pass
    try:
        datetime_stats['range'] = str(ut.unixtime_to_timedelta(int(round(unixtime_stats['max'] - unixtime_stats['min']))))
    except KeyError:
        pass
    return datetime_stats


#def datetime_to_posixtime(dt):
#    return dt.toordinal()


if __name__ == '__main__':
    """
    CommandLine:
        python -c "import utool, utool.util_time; utool.doctest_funcs(utool.util_time, allexamples=True)"
        python -c "import utool, utool.util_time; utool.doctest_funcs(utool.util_time)"
        python -m utool.util_time
        python -m utool.util_time --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
