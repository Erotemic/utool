# -*- coding: utf-8 -*-
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
import calendar
import datetime
from utool import util_inject
from utool import util_cplat
from utool import util_arg
from timerit import Timerit as RealTimerit
print, rrr, profile = util_inject.inject2(__name__)
print_ = util_inject.make_module_write_func(__name__)


if util_cplat.WIN32 and six.PY2:
    # Use time.clock in win32
    default_timer = time.clock
else:
    default_timer = time.time


# --- Timing ---
def tic(msg=None):
    """
    similar to matlab tic

    SeeAlso:
        ut.toc
    """
    return (msg, default_timer())


def toc(tt, return_msg=False, write_msg=True, verbose=None):
    """
    similar to matlab toc

    SeeAlso:
        ut.tic
    """
    if verbose is not None:
        write_msg = verbose
    (msg, start_time) = tt
    ellapsed = (default_timer() - start_time)
    if (not return_msg) and write_msg and msg is not None:
        sys.stdout.write('...toc(%.4fs, ' % ellapsed + '"' + str(msg) + '"' + ')\n')
    if return_msg:
        return msg
    else:
        return ellapsed


def get_timestamp(format_='iso', use_second=False, delta_seconds=None,
                  isutc=False, timezone=False):
    """
    get_timestamp

    Args:
        format_ (str): (tag, printable, filename, other)
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
    # TODO: time.timezone
    if format_ == 'int':
        if isutc:
            stamp = int(time.mktime(time.gmtime()))
        else:
            stamp = int(time.mktime(time.localtime()))
        return stamp
    if isutc:
        now = datetime.datetime.utcnow()
    else:
        now = datetime.datetime.now()
    if delta_seconds is not None:
        now += datetime.timedelta(seconds=delta_seconds)
    if format_ == 'iso':
        # ISO 8601
        #utcnow = datetime.datetime.utcnow()
        #utcnow.isoformat()
        localOffsetHour = time.timezone // 3600
        utc_offset = '-' + str(localOffsetHour) if localOffsetHour < 0 else '+' + str(localOffsetHour)
        stamp = time.strftime('%Y-%m-%dT%H%M%S') + utc_offset
        return stamp
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
    if timezone:
        if isutc:
            stamp += '_UTC'
        else:
            from pytz import reference
            localtime = reference.LocalTimezone()
            tzname = localtime.tzname(now)
            stamp += '_' + tzname
    return stamp


def get_datestamp(explicit=True, isutc=False):
    if isutc:
        now = datetime.datetime.utcnow()
    else:
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
        >>> import utool as ut
        >>> timer = ut.Timer('Timer test!', verbose=1)
        >>> with timer:
        >>>     prime = ut.get_nth_prime(400)
        >>> assert timer.ellapsed > 0
    """
    def __init__(self, msg='', verbose=True, newline=True):
        self.msg = msg
        self.verbose = verbose
        self.newline = newline
        self.tstart = -1
        self.ellapsed = -1

    def tic(self):
        if self.verbose:
            sys.stdout.flush()
            print_('\ntic(%r)' % self.msg)
            if self.newline:
                print_('\n')
            sys.stdout.flush()
        self.tstart = default_timer()

    def toc(self):
        ellapsed = (default_timer() - self.tstart)
        if self.verbose:
            print_('...toc(%r)=%.4fs\n' % (self.msg, ellapsed))
            sys.stdout.flush()
        return ellapsed

    start = tic
    stop = toc

    def __enter__(self):
        self.tic()
        return self

    def __exit__(self, type_, value, trace):
        self.ellapsed = self.toc()
        if trace is not None:
            return False  # return a falsey value on error


class Timerit(RealTimerit):
    """
    Reports the average time to run a block of code.

    Unlike `timeit`, `Timerit` can handle multiline blocks of code

    Args:
        num (int): number of times to run the loop
        label (str): identifier for printing
        unit (str): reporting unit of time (e.g. 'ms', 's', None)

    CommandLine:
        python -m utool.util_time Timerit
        python -m utool.util_time Timerit:0
        python -m utool.util_time Timerit:1

    Notes:
        Minimal syntax with less-precise timing
        ```python
        import utool as ut
        for timer in ut.Timerit(100):
            # <write code to time here>
        ```

        Full syntax with most-precise timing
        ```python
        import utool as ut
        for timer in ut.Timerit(100):
            # <write untimed setup code here>
            with timer:
                # <write code to time here>
        ```

        Can also keep track of the Timerit object for extra statistics
        ```python
        import utool as ut
        t1 = ut.Timerit(100)
        for timer in t1:
            # <write untimed setup code here>
            with timer:
                # <write code to time here>
        # <you can now access Timerit attributes like t1.total_time>
        ```

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> num = 15
        >>> t1 = ut.Timerit(num)
        >>> for timer in t1:
        >>>     # <write untimed setup code here> this example has no setup
        >>>     with timer:
        >>>         # <write code to time here> for example...
        >>>         ut.get_nth_prime_bruteforce(100)
        >>> # <you can now access Timerit attributes>
        >>> print('t1.total_time = %r' % (t1.total_time,))
        >>> assert t1.total_time > 0
        >>> assert t1.n_loops == t1.num
        >>> assert t1.n_loops == num

    Example:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> num = 10000
        >>> n = 50
        >>> # If the timer object is unused, time will still be recoreded,
        >>> # but with less precision.
        >>> for _ in ut.Timerit(num, 'inprecise'):
        >>>     ut.get_nth_prime_bruteforce(n)
        >>> # Using the timer object results in the most precise timeings
        >>> for timer in ut.Timerit(num, 'precise'):
        >>>     with timer:
        >>>         ut.get_nth_prime_bruteforce(n)
    """
    ...
    # def __init__(self, num, label=None, unit=None, verbose=1):
    #     self.num = num
    #     self.label = label
    #     self.times = []
    #     self.verbose = verbose
    #     self.total_time = None
    #     self.n_loops = None
    #     self.unit = unit

    # def __iter__(self):
    #     import utool as ut
    #     if self.verbose > 1:
    #         if self.label is None:
    #             print('Timing for %d loops' % self.num)
    #         else:
    #             print('Timing %s for %d loops.' % (self.label, self.num,))
    #     self.n_loops = 0
    #     self.total_time = 0
    #     # Create a foreground and background timer
    #     bg_timer = ut.Timer(verbose=0)   # (ideally this is unused)
    #     fg_timer = ut.Timer(verbose=0)   # (used directly by user)
    #     # Core timing loop
    #     for i in range(self.num):
    #         # Start background timer (in case the user doesnt use fg_timer)
    #         # Yield foreground timer to let the user run a block of code
    #         # When we return from yield the user code will have just finishec
    #         # Then record background time + loop overhead
    #         bg_timer.tic()
    #         yield fg_timer
    #         bg_time = bg_timer.toc()
    #         # Check if the fg_timer object was used, but fallback on bg_timer
    #         if fg_timer.ellapsed >= 0:
    #             block_time = fg_timer.ellapsed  # high precision
    #         else:
    #             block_time = bg_time  # low precision
    #         # record timeings
    #         self.times.append(block_time)
    #         self.total_time += block_time
    #         self.n_loops += 1
    #     # Timeing complete, print results
    #     assert len(self.times) == self.num, 'incorrectly recorded times'
    #     if self.verbose > 0:
    #         self._print_report(self.verbose)

    # @property
    # def ave_secs(self):
    #     return self.total_time / self.n_loops

    # def _print_report(self, verbose=1):
    #     import utool as ut
    #     ave_secs = self.ave_secs
    #     if self.label is None:
    #         print('Timing complete, %d loops' % (self.n_loops,))
    #     else:
    #         print('Timing complete for: %s, %d loops' % (self.label,
    #                                                      self.n_loops))
    #     if verbose > 2:
    #         body = ut.second_str(self.total_time, unit=self.unit, precision=4)
    #         print('    body took: %s' % body)
    #     perloop = ut.second_str(ave_secs, unit=self.unit, precision=4)
    #     print('    time per loop : %s' % (perloop,))


def determine_timestamp_format(datetime_str, warn=True):
    r"""
    Args:
        datetime_str (str):

    Returns:
        str:

    References:
        https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior

    CommandLine:
        python -m utool.util_time --exec-determine_timestamp_format

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> import utool as ut
        >>> datetime_str_list = [
        >>>     '0000:00:00 00:00:00',
        >>>     '    :  :     :  :  ',
        >>>     '2015:04:01 00:00:00',
        >>>     '2080/04/01 00:00:00',
        >>>     '2005-10-27T14:35:20+02:00',
        >>>     '6:35:01\x002006:03:19 1',
        >>>     '2016/05/03 16:34:57 EST'
        >>> ]
        >>> result = ut.repr4([determine_timestamp_format(datetime_str)
        >>>            for datetime_str in datetime_str_list])
        >>> print(result)
    """
    import re
    # try to determine the format
    clean_datetime_str = datetime_str.replace('\x00', ' ').strip(';').strip()
    if len(clean_datetime_str) == 25 and 'T' in clean_datetime_str:
        # Delete last colon from ISO 8601 format
        # clean_datetime_str = clean_datetime_str[:-3] + clean_datetime_str[-2:]
        if True or six.PY2:
            if warn:
                print('WARNING: Python 2.7 does not support %z directive '
                      'in strptime, ignoring timezone in parsing: ' +
                      clean_datetime_str)
            clean_datetime_str = clean_datetime_str[:-6]

    year_regex  = r'(\d\d)?\d\d'
    month_regex = '[0-1]?[0-9]'
    day_regex   = '[0-3]?[0-9]'

    time_regex = r'[0-6]?[0-9]:[0-6]?[0-9]:[0-6]?[0-9]'

    #odd_time_regex = r'[0-6]?[0-9]:[0-6]?[0-9]:[0-6 ]?[0-9]'

    date_regex1 = '/'.join([year_regex, month_regex, day_regex])
    date_regex2 = ':'.join([year_regex, month_regex, day_regex])
    date_regex3 = '-'.join([year_regex, month_regex, day_regex])
    datetime_regex1 = date_regex1 + ' ' + time_regex
    datetime_regex2 = date_regex2 + ' ' + time_regex
    datetime_regex3 = date_regex3 + 'T' + time_regex  # + r'\+[0-2]?[0-9]?[0-6]?[0-9]'
    datetime_regex4 = time_regex + ' ' + date_regex2 + ' 1'

    timefmt = None

    if re.match(datetime_regex1, clean_datetime_str):
        timefmt = '%Y/%m/%d %H:%M:%S'
    elif re.match(datetime_regex2, clean_datetime_str):
        timefmt = '%Y:%m:%d %H:%M:%S'
    elif re.match(datetime_regex3, clean_datetime_str):
        # timefmt = '%Y-%m-%dT%H:%M:%S%z'
        timefmt = '%Y-%m-%dT%H:%M:%S'
    elif re.match(datetime_regex4, clean_datetime_str):
        # timefmt = '%Y-%m-%dT%H:%M:%S%z'
        timefmt = '%H:%M:%S %Y:%m:%d 1'
    # Just dont accept this bad format
    #elif re.match(datetime_regex3, clean_datetime_str):
    #    timefmt = '%Y:%m:%d %H:%M: %S'
    else:
        if isinstance(clean_datetime_str, six.string_types):
            if len(clean_datetime_str.strip()) == 0:
                return None
            elif len(clean_datetime_str.strip(':/ ')) == 0:
                return None
            elif clean_datetime_str.find('No EXIF Data') == 0:
                return None
            elif clean_datetime_str.find('Invalid') == 0:
                return None
            elif clean_datetime_str == '0000:00:00 00:00:00':
                return None
            elif [ ord(_) >= 128 for _ in clean_datetime_str ].count(True) > 1:
                return None
        #return -1
        #import utool as ut
        #ut.embed()
        msg = 'Unknown format: datetime_str=%r' % (datetime_str,)
        print(msg)
        return None
        #raise NotImplementedError(msg)
    return timefmt


def local_timezone():
    import time
    return time.tzname


def utcnow_tz():
    import pytz
    dt = datetime.datetime.utcnow()
    dt = dt.replace(tzinfo=pytz.timezone('UTC'))
    return dt


def parse_timestamp(timestamp, zone='UTC', timestamp_format=None):
    r"""
    pip install delorean

    Args:
        timestamp (str): timestampe string
        zone (bool): assumes input is zone (only if not specified) and gives
            output in zone.

    CommandLine:
        python -m utool.util_time --test-parse_timestamp
        python -m utool.util_time parse_timestamp

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> import utool as ut
        >>> utc = True
        >>> timestampe_format = None
        >>> timestamps = [
        >>>     ('2015:04:01 00:00:00',),
        >>>     ('2005-10-27T14:35:20+02:00',),
        >>>     ('2000-01-01T09:00:00-05:00', 'UTC'),
        >>>     ('2000-01-01T09:00:00-05:00', 'EST'),
        >>>     ('2000-01-01T09:00:00', 'EST'),
        >>>     ('2000-01-01T09:00:00', 'UTC'),
        >>>     ('6:35:01\x002006:03:19 1',),
        >>>     ('2016/08/18 10:51:02 EST',),
        >>>     ('2016-08-18T10:51:02-05:00',),
        >>> ]
        >>> timestamp = timestamps[-1][0]
        >>> dn_list = [parse_timestamp(*args) for args in timestamps]
        >>> result = ut.NEWLINE.join([str(dn) for dn in dn_list])
        >>> print(result)
        2015-04-01 00:00:00+00:00
        2005-10-27 12:35:20+00:00
        2000-01-01 14:00:00+00:00
        2000-01-01 09:00:00-05:00
        2000-01-01 09:00:00-05:00
        2000-01-01 09:00:00+00:00
        2006-03-19 06:35:01+00:00
        2016-08-18 15:51:02+00:00
        2016-08-18 15:51:02+00:00
    """
    if timestamp is None:
        return None

    use_delorean = True or six.PY2
    if use_delorean:
        import delorean
        ## customize delorean string method
        #def __str__(self):
        #    return str(self.datetime)
        #    #return str(self.datetime) + ' ' + str(self.timezone)
        #delorean.Delorean.__str__ = __str__
        ## method types must be injected into the class
        ##ut.inject_func_as_method(dn, __str__, '__repr__', override=True)

    if not isinstance(timestamp, six.string_types):
        raise NotImplementedError('Unknown format: timestamp=%r' % (timestamp,))

    # Normal format, or non-standard year first data
    if timestamp_format is None:
        # dont warn because we will take care of utc
        timefmt = determine_timestamp_format(timestamp, warn=False)
    else:
        timefmt = timestamp_format
    if timefmt is None or not isinstance(timefmt, six.string_types):
        raise AssertionError('unknown timestamp_format=%r' % (timestamp_format,))

    # Fixup timestamp
    utc_offset = None
    if len(timestamp) == 20 and '\x00' in timestamp:
        timestamp_ = timestamp.replace('\x00', ' ').strip(';').strip()
    elif use_delorean and len(timestamp) > 19:
        timestamp_ = timestamp[:19].strip(';').strip()
        utc_offset = timestamp[19:]
    else:
        timestamp_ = timestamp

    dt_ = datetime.datetime.strptime(timestamp_, timefmt)
    if use_delorean:
        #if utc and utc_offset is not None:
        #if utc:
        #    dn_ = delorean.Delorean(dt_, 'UTC')
        #else:
        if zone is None:
            zone = time.tzname[0]
        if zone == 'local':
            zone = time.tzname[0]
        dn_ = delorean.Delorean(dt_, zone)
    else:
        dn_ = dt_

    if utc_offset is not None and zone == 'UTC':
        if use_delorean:
            # Python 2.7 does not account for timezones
            if ':' in utc_offset:
                sign = {' ': +1, '+': +1, '-': -1}[utc_offset[0]]
                hours, seconds = utc_offset[1:].split(':')
                delta_ = datetime.timedelta(hours=int(hours), seconds=int(seconds))
                delta = sign * delta_
            else:
                import pytz
                tzname = utc_offset.strip()
                delta = pytz.timezone(tzname).utcoffset(dt_)
            # Move back to utc
            dn = dn_ - delta
        else:
            raise AssertionError('python3 should take care of timezone')
    else:
        dn = dn_

    if use_delorean:
        if not zone != 'UTC':
            dn.shift(zone)
        return dn.datetime
    #return dn


def exiftime_to_unixtime(datetime_str, timestamp_format=None, strict=None):
    r"""
    converts a datetime string to posixtime (unixtime)
    Use parse timestamp instead

    Args:
        datetime_str     (str):
        timestamp_format (int):

    Returns:
        int: unixtime seconds from 1970 (currently not UTC; this will change)

    CommandLine:
        python -m utool.util_time --test-exiftime_to_unixtime:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> datetime_str = '0000:00:00 00:00:00'
        >>> timestamp_format = 1
        >>> result = exiftime_to_unixtime(datetime_str, timestamp_format)
        >>> print(result)
        -1

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> datetime_str = '2015:04:01 00:00:00'
        >>> timestamp_format = 1
        >>> result = exiftime_to_unixtime(datetime_str, timestamp_format)
        >>> print(result)
        1427846400

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> datetime_str = '2005-10-27T14:35:20+02:00'
        >>> timestamp_format = None
        >>> result = exiftime_to_unixtime(datetime_str, timestamp_format)
        >>> print(result)
        1130423720

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> datetime_str = '6:35:01\x002006:03:19 1'
        >>> timestamp_format = None
        >>> result = exiftime_to_unixtime(datetime_str, timestamp_format)
        >>> print(result)
        1142750101
    """
    invalid_value = -1
    if isinstance(datetime_str, int):
        if datetime_str == -1:
            return invalid_value
    elif datetime_str is None:
        return None

    # TODO: use parse_timestamp to reduce duplicate code
    #try:
    #    dt = parse_timestamp(datetime_str)
    #except TypeError:
    #    #if datetime_str is None:
    #        #return -1
    #    return -1
    #except ValueError as ex:
    #    if strict is None:
    #        ...

    if not isinstance(datetime_str, six.string_types):
        raise NotImplementedError('Unknown format: datetime_str=%r' % (datetime_str,))
    # Normal format, or non-standard year first data
    if timestamp_format is None:
        timefmt = determine_timestamp_format(datetime_str)
        if timefmt is None:
            return invalid_value
    elif timestamp_format == 2:
        timefmt = '%m/%d/%Y %H:%M:%S'
    elif timestamp_format == 1:
        timefmt = '%Y:%m:%d %H:%M:%S'
    else:
        assert isinstance(timestamp_format, six.string_types)
        timefmt = timestamp_format
        #raise AssertionError('unknown timestamp_format=%r' % (timestamp_format,))
    try:
        if len(datetime_str) == 20 and '\x00' in datetime_str:
            datetime_str_ = datetime_str.replace('\x00', ' ').strip(';').strip()
        elif len(datetime_str) > 19:
            datetime_str_ = datetime_str[:19].strip(';').strip()
        else:
            datetime_str_ = datetime_str
        #try:
        dt = datetime.datetime.strptime(datetime_str_, timefmt)
        #except ValueError as ex:
        #    import utool as ut
        #    ut.printex(ex, iswarning=True)
        #    return invalid_value
        return calendar.timegm(dt.timetuple())
        #return time.mktime(dt.timetuple())
    except TypeError:
        #if datetime_str is None:
        #    return -1
        return -1
    except ValueError as ex:
        if strict is None:
            strict = util_arg.SUPER_STRICT
        #strict = False
        #from utool.util_arg import STRICT
        if isinstance(datetime_str, six.string_types):
            if len(datetime_str_.strip()) == 0:
                return invalid_value
            if datetime_str_.find('No EXIF Data') == 0:
                return invalid_value
            if datetime_str_.find('Invalid') == 0:
                return invalid_value
            if datetime_str_ == '0000:00:00 00:00:00':
                return invalid_value
        print('<!!! ValueError !!!>')
        print('[util_time] Caught Error: ' + repr(ex))
        print('[util_time] type(datetime_str)  = %r' % type(datetime_str))
        print('[util_time] repr(datetime_str)  = %r' % datetime_str)
        print('[util_time]     (datetime_str)  = %s' % datetime_str)
        print('[util_time]  len(datetime_str)  = %d' % len(datetime_str))
        print('[util_time] repr(datetime_str_) = %r' % datetime_str_)
        print('[util_time]  len(datetime_str_) = %d' % len(datetime_str_))
        print('</!!! ValueError !!!>')

        debug = True
        if debug:
            def find_offending_part(datetime_str_, timefmt, splitchar=' '):
                import utool as ut
                parts_list = datetime_str_.split(splitchar)
                fmt_list = timefmt.split(splitchar)
                if len(parts_list) == 1:
                    return
                for part, fmt in zip(parts_list, fmt_list):
                    print('Trying:')
                    with ut.Indenter('  '):
                        try:
                            print('fmt = %r' % (fmt,))
                            print('part = %r' % (part,))
                            datetime.datetime.strptime(part, fmt)
                        except ValueError:
                            find_offending_part(part, fmt, '/')
                            print('Failed')
                        else:
                            print('Passed')
            find_offending_part(datetime_str_, timefmt)

        #import utool as ut
        #ut.embed()
        if strict:
            raise
        else:
            print('Supressed ValueError')
            return invalid_value


def parse_timedelta_str(str_):
    r"""
    Args:
        str_ (str):

    Returns:
        float: timedelta

    CommandLine:
        python -m utool.util_time --exec-parse_timedelta_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> str_ = '24h'
        >>> timedelta = parse_timedelta_str(str_)
        >>> result = ('timedelta = %s' % (str(timedelta),))
        >>> print(result)
        timedelta = 86400.0
    """
    if str_.endswith('m'):
        timedelta = float(str_[0:-1]) * 60
    elif str_.endswith('h'):
        timedelta = float(str_[0:-1]) * 60 * 60
    elif str_.endswith('s'):
        timedelta = float(str_[0:-1])
    else:
        raise NotImplementedError('Unknown timedelta format %r' % (str_))
    return timedelta


def ensure_timedelta(str_or_num):
    if isinstance(str_or_num, six.string_types):
        return parse_timedelta_str(str_or_num)
    else:
        return str_or_num


def unixtime_to_datetimestr(unixtime, timefmt='%Y/%m/%d %H:%M:%S', isutc=True):
    """
    TODO: ranme to datetimestr
    """
    try:
        if unixtime == -1:
            return 'NA'
        if unixtime is None:
            return None
        if isutc:
            return datetime.datetime.utcfromtimestamp(float(unixtime)).strftime(timefmt)
        else:
            return datetime.datetime.fromtimestamp(float(unixtime)).strftime(timefmt)
    except ValueError:
        raise
        #return 'NA'


def unixtime_to_datetimeobj(unixtime, isutc=True):
    try:
        if unixtime == -1:
            return 'NA'
        if unixtime is None:
            return None
        if isutc:
            return datetime.datetime.utcfromtimestamp(float(unixtime))
        else:
            return datetime.datetime.fromtimestamp(float(unixtime))
    except ValueError:
        raise


def unixtime_to_timedelta(unixtime_diff):
    """ alias for get_unix_timedelta """
    return get_unix_timedelta(unixtime_diff)


def get_unix_timedelta(unixtime_diff):
    timedelta = datetime.timedelta(seconds=abs(unixtime_diff))
    return timedelta


def get_unix_timedelta_str(unixtime_diff):
    """
    TODO: rectify this function with get_posix_timedelta_str

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
        >>> import utool as ut
        >>> unixtime_diff = 0
        >>> timestr = get_unix_timedelta_str(unixtime_diff)
        >>> timestr_list = [get_unix_timedelta_str(_) for _ in [-9001, -1, 0, 1, 9001]]
        >>> result = ut.repr2(timestr_list)
        >>> print(result)
        ['2 hours 30 minutes 1 second', '1 second', '0 seconds', '1 second', '2 hours 30 minutes 1 second']
    """
    timedelta = get_unix_timedelta(unixtime_diff)
    timestr = get_timedelta_str(timedelta)
    return timestr


def get_timedelta_str(timedelta, exclude_zeros=False):
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

    def append_cases(unit, fmtlbl):
        if not exclude_zeros or unit != 0:
            if unit == 1:
                fmtstr_list.append('{%s} %s' % (fmtlbl, fmtlbl))
            else:
                fmtstr_list.append('{%s} %ss' % (fmtlbl, fmtlbl))
            fmtdict[fmtlbl] = unit

    if abs(days) > 0:
        append_cases(days, 'day')
    if len(fmtstr_list) > 0 or abs(hours) > 0:
        append_cases(hours, 'hour')
    if len(fmtstr_list) > 0 or abs(minutes) > 0:
        append_cases(minutes, 'minute')
    if len(fmtstr_list) > 0 or abs(seconds) > 0:
        append_cases(seconds, 'second')
    fmtstr = ' '.join(fmtstr_list)
    timedelta_str = fmtstr.format(**fmtdict)
    return timedelta_str


def get_posix_timedelta_str(posixtime, year=False, approx=True):
    """
    get_timedelta_str

    TODO: rectify this function with get_unix_timedelta_str (unix_timedelta_str probably has better implementation)

    Returns:
        str: timedelta_str, formated time string

    CommandLine:
        python -m utool.util_time --test-get_posix_timedelta_str

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> import utool as ut
        >>> posixtime_list = [-13, 10.2, 10.2 ** 2, 10.2 ** 3, 10.2 ** 4, 10.2 ** 5, 10.2 ** 8, 60 * 60 * 60 * 24 * 7]
        >>> posixtime = posixtime_list[-1]
        >>> timedelta_str = [get_posix_timedelta_str(posixtime) for posixtime in posixtime_list]
        >>> result = ut.repr2(timedelta_str)
        >>> print(result)
        ['-00:00:13', '00:00:10', '00:01:44', '00:17:41', '03:00:24', '1 day', '193 weeks', '60 weeks']

    Timeit::
        import datetime
        # Seems like like timedelta is just faster. must be because it is builtin
        %timeit get_posix_timedelta_str(posixtime)
        %timeit str(datetime.timedelta(seconds=posixtime))

    """
    import numpy as np
    if np.isnan(posixtime):
        return 'NaN'
    sign, posixtime_ = (1, posixtime) if posixtime >= 0 else (-1, -posixtime)
    seconds_, subseconds = divmod(posixtime_, 1)
    minutes_, seconds    = divmod(int(seconds_), 60)
    hours_, minutes      = divmod(minutes_, 60)
    days_, hours         = divmod(hours_, 24)
    weeks_, days         = divmod(days_, 7)
    if year:
        years, weeks = divmod(weeks_, 52)  # not accurate
    else:
        years = 0
        weeks = weeks_
    timedelta_parts = []
    if subseconds > 0:
        timedelta_parts += [('%.2f' % (subseconds,))[1:]]
    timedelta_parts += [':'.join(['%02d' % _ for _ in (hours, minutes, seconds)])]
    import utool as ut
    if days > 0:
        timedelta_parts += ['%d %s ' % (days, ut.pluralize('day', days))]
    if weeks > 0:
        timedelta_parts += ['%d %s ' % (weeks, ut.pluralize('week', weeks))]
    if years > 0:
        timedelta_parts += ['%d %s ' % (years, ut.pluralize('year', years))]
    if sign == -1:
        timedelta_parts += ['-']
    else:
        timedelta_parts += ['']
    if approx is not False:
        if approx is True:
            approx = 1
        timedelta_str = ''.join(timedelta_parts[::-1][0:(approx + 1)]).strip()
    else:
        timedelta_str = ''.join(timedelta_parts[::-1])
    return timedelta_str


def get_posix_timedelta_str2(posixtime):
    try:
        return get_posix_timedelta_str(posixtime)
    except ValueError:
        # handle nones and nans
        return 'None'


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


#def get_month():
#    return datetime.datetime.now().month


#def get_day():
#    return datetime.datetime.now().day


#def get_year():
#    return datetime.datetime.now().year


def get_timestats_str(unixtime_list, newlines=1, full=True, isutc=True):
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
        >>> unixtime_list = [0, 0 + 60 * 60 * 5 , 10 + 60 * 60 * 5, 100 + 60 * 60 * 5, 1000 + 60 * 60 * 5]
        >>> newlines = 1
        >>> full = False
        >>> timestat_str = get_timestats_str(unixtime_list, newlines, full=full, isutc=True)
        >>> result = ut.align(str(timestat_str), ':')
        >>> print(result)
        {
            'max'  : '1970/01/01 05:16:40',
            'mean' : '1970/01/01 04:03:42',
            'min'  : '1970/01/01 00:00:00',
            'range': '5:16:40',
            'std'  : '2:02:01',
        }

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_time import *  # NOQA
        >>> import utool as ut
        >>> unixtime_list = [0, 0 + 60 * 60 * 5 , 10 + 60 * 60 * 5, 100 + 60 * 60 * 5, 1000 + 60 * 60 * 5, float('nan'), 0]
        >>> newlines = 1
        >>> timestat_str = get_timestats_str(unixtime_list, newlines, isutc=True)
        >>> result = ut.align(str(timestat_str), ':')
        >>> print(result)
        {
            'max'    : '1970/01/01 05:16:40',
            'mean'   : '1970/01/01 03:23:05',
            'min'    : '1970/01/01 00:00:00',
            'nMax'   : 1,
            'nMin'   : 2,
            'num_nan': 1,
            'range'  : '5:16:40',
            'shape'  : (7,),
            'std'    : '2:23:43',
        }

    """
    import utool as ut
    datetime_stats = get_timestats_dict(unixtime_list, full=full, isutc=isutc)
    timestat_str = ut.repr4(datetime_stats, newlines=newlines)
    return timestat_str


def get_timestats_dict(unixtime_list, full=True, isutc=True):
    import utool as ut
    unixtime_stats = ut.get_stats(unixtime_list, use_nan=True)
    datetime_stats = {}
    if unixtime_stats.get('empty_list', False):
        datetime_stats = unixtime_stats
        return datetime_stats
    # elif unixtime_stats.get('num_nan', 0) == unixtime_stats.get('shape')[0]:
    #     datetime_stats = unixtime_stats
    #     return datetime_stats
    for key in ['max', 'min', 'mean']:
        try:
            datetime_stats[key] = ut.unixtime_to_datetimestr(unixtime_stats[key], isutc=isutc)
        except KeyError:
            pass
        except (ValueError, OSError):
            datetime_stats[key]  = 'NA'
        except Exception as ex:
            ut.printex(ex, keys=['key', 'unixtime_stats'])
            raise
    for key in ['std']:
        try:
            datetime_stats[key] = str(ut.get_unix_timedelta(int(round(unixtime_stats[key]))))
        except KeyError:
            pass
        except (ValueError, OSError):
            datetime_stats[key]  = 'NA'
    try:
        datetime_stats['range'] = str(ut.get_unix_timedelta(int(round(unixtime_stats['max'] - unixtime_stats['min']))))
    except KeyError:
        pass
    except (ValueError, OSError):
        datetime_stats['range']  = 'NA'

    if full:
        #unused_keys = (set(unixtime_stats.keys()) - set(datetime_stats.keys())
        for key in ['nMax', 'num_nan', 'shape', 'nMin']:
            datetime_stats[key] = unixtime_stats[key]
    #print('Unused keys = %r' % (set(unixtime_stats.keys()) - set(datetime_stats.keys()),))
    return datetime_stats


def date_to_datetime(date, fraction=0.0):
    """
    fraction is how much through the day you are. 0=start of the day, 1=end of the day.
    """
    day_seconds = (60 * 60 * 24) - 1
    total_seconds = int(day_seconds * fraction)
    delta = datetime.timedelta(seconds=total_seconds)
    time = datetime.time()
    dt = datetime.datetime.combine(date, time) + delta
    return dt


def datetime_to_posixtime(dt):
    posixtime = time.mktime(dt.timetuple())
    return posixtime


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
