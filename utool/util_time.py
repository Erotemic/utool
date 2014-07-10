from __future__ import absolute_import, division, print_function
import sys
import time
import datetime
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[time]')


# --- Timing ---
def tic(msg=None):
    return (msg, time.time())


def toc(tt, return_msg=False, write_msg=True):
    (msg, start_time) = tt
    ellapsed = (time.time() - start_time)
    if not return_msg and write_msg and msg is not None:
        sys.stdout.write('...toc(%.4fs, ' % ellapsed + '"' + str(msg) + '"' + ')\n')
    if return_msg:
        return msg
    else:
        return ellapsed


def get_timestamp(format_='filename', use_second=False):
    now = datetime.datetime.now()
    if format_ == 'tag':
        time_tup = (now.year - 2000, now.month, now.day)
        stamp = '%02d%02d%02d' % time_tup
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


class Timer(object):
    """
    Timer with-statment context object
    e.g with Timer() as t: some_function()
    A ++quality utool
    """
    def __init__(self, msg='', verbose=True, newline=True):
        self.msg = msg
        self.verbose = verbose
        self.newline = newline
        self.tstart = -1
        self.ellapsed = -1
        self.tic()

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
        if self.msg is not None:
            sys.stdout.write('---tic---' + self.msg + '  \n')
        self.tic()
        return self

    def __exit__(self, type_, value, trace):
        self.ellapsed = self.toc()
        if trace is not None:
            print('[util_time] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
        #return self.ellapsed


def exiftime_to_unixtime(datetime_str, timestamp_format=1):
    try:
        # Normal format, or non-standard year first data
        if timestamp_format == 2:
            timefmt = '%m/%d/%Y %H:%M:%S'
        else:
            timefmt = '%Y:%m:%d %H:%M:%S'
        if len(datetime_str) > 19:
            datetime_str = datetime_str[0:20].strip(";")
        dt = datetime.datetime.strptime(datetime_str, timefmt)
        return time.mktime(dt.timetuple())
    except TypeError:
        #if datetime_str is None:
            #return -1
        return -1
    except ValueError as ex:
        if isinstance(datetime_str, (str, unicode)):
            if datetime_str.find('No EXIF Data') == 0:
                return -1
            if datetime_str.find('Invalid') == 0:
                return -1
            if datetime_str == '0000:00:00 00:00:00':
                return -1
        print('!!!!!!!!!!!!!!!!!!')
        print('[util_time] Caught Error: ' + repr(ex))
        print('[util_time] type(datetime_str) = %r' % type(datetime_str))
        print('[util_time] datetime_str = %r' % datetime_str)
        print('[util_time] datetime_str = %s' % datetime_str)
        raise


def unixtime_to_datetime(unixtime, timefmt='%Y/%m/%d %H:%M:%S'):
    if unixtime == -1:
        return 'NA'
    return datetime.datetime.fromtimestamp(unixtime).strftime(timefmt)


def get_unix_timedelta(unixtime_diff):
    timedelta = datetime.timedelta(seconds=abs(unixtime_diff))
    return timedelta


def get_month():
    return datetime.datetime.now().month


def get_day():
    return datetime.datetime.now().day


def get_year():
    return datetime.datetime.now().year
