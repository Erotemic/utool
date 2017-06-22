# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from utool import util_inject
from utool import util_str
print, rrr, profile = util_inject.inject2(__name__)

try:
    # Resource does not exist in win32
    import resource

    def time_in_usermode():
        stime = resource.getrusage(resource.RUSAGE_SELF).ru_stime
        return stime

    def time_in_systemmode():
        utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        return utime

    def peak_memory():
        """Returns the resident set size (the portion of
        a process's memory that is held in RAM.)
        """
        # MAXRSS is expressed in kilobytes. Convert to bytes
        # FIXME: MAXRSS is NOT expressed in kilobytes. use resource.getpagesize to Convert to bytes
        # References: http://stackoverflow.com/questions/938733/total-memory-used-by-python-process
        #resource.getpagesize
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return maxrss

    def get_resource_limits():
        #rlimit_keys = [key for key in six.iterkeys(resource.__dict__) if key.startswith('RLIMIT_')]
        #print('\n'.join(['(\'%s\', resource.%s),' % (key.replace('RLIMIT_', ''), key) for key in rlimit_keys]))
        rlim_keytups = [
            ('MEMLOCK', resource.RLIMIT_MEMLOCK),
            ('NOFILE', resource.RLIMIT_NOFILE),
            ('CPU', resource.RLIMIT_CPU),
            ('DATA', resource.RLIMIT_DATA),
            ('OFILE', resource.RLIMIT_OFILE),
            ('STACK', resource.RLIMIT_STACK),
            ('FSIZE', resource.RLIMIT_FSIZE),
            ('CORE', resource.RLIMIT_CORE),
            ('NPROC', resource.RLIMIT_NPROC),
            ('AS', resource.RLIMIT_AS),
            ('RSS', resource.RLIMIT_RSS),
        ]
        rlim_valtups = [(lbl, resource.getrlimit(rlim_key)) for (lbl, rlim_key) in rlim_keytups]
        def rlimval_str(rlim_val):
            soft, hard = rlim_val
            softstr = util_str.byte_str2(soft) if soft != -1 else 'None'
            hardstr = util_str.byte_str2(hard) if hard != -1 else 'None'
            return '%12s, %12s' % (softstr, hardstr)
        rlim_strs = ['%8s: %s' % (lbl, rlimval_str(rlim_val)) for (lbl, rlim_val) in rlim_valtups]
        print('Resource Limits: ')
        print('%8s  %12s  %12s' % ('id', 'soft', 'hard'))
        print('\n'.join(rlim_strs))
        return rlim_strs

    #def rusage_flags():
        #0	ru_utime	time in user mode (float)
        #1	ru_stime	time in system mode (float)
        #2	ru_maxrss	maximum resident set size
        #3	ru_ixrss	shared memory size
        #4	ru_idrss	unshared memory size
        #5	ru_isrss	unshared stack size
        #6	ru_minflt	page faults not requiring I/O
        #7	ru_majflt	page faults requiring I/O
        #8	ru_nswap	number of swap outs
        #9	ru_inblock	block input operations
        #10	ru_oublock	block output operations
        #11	ru_msgsnd	messages sent
        #12	ru_msgrcv	messages received
        #13	ru_nsignals	signals received
        #14	ru_nvcsw	voluntary context switches
        #15	ru_nivcsw	involuntary context switches
except ImportError:
    def time_in_usermode():
        raise NotImplementedError('unavailable in win32')

    def time_in_systemmode():
        raise NotImplementedError('unavailable in win32')

    def peak_memory():
        """Returns the resident set size (the portion of
        a process's memory that is held in RAM.)
        """
        raise NotImplementedError('unavailable in win32')

    def get_resource_limits():
        raise NotImplementedError('unavailable in win32')


def time_str2(seconds):
    return '%.2f sec' % (seconds,)


def print_resource_usage():
    print(get_resource_usage_str())


def get_resource_usage_str():
    usage_str_list = [
        ('+______________________'),
        ('|    RESOURCE_USAGE    process.get_memory_info()[0] / float(2 ** 20) '),
        ('|  * current_memory = %s' % util_str.byte_str2(current_memory_usage())),
    ]
    try:
        usage_str_list.extend([
            ('|  * peak_memory    = %s' % util_str.byte_str2(peak_memory())),
            ('|  * user_time      = %s' % time_str2(time_in_usermode())),
            ('|  * system_time    = %s' % time_str2(time_in_systemmode())),
        ])
    except Exception:
        pass
    usage_str_list.append('L______________________')
    usage_str = '\n'.join(usage_str_list)
    return usage_str


def current_memory_usage():
    """
    Returns this programs current memory usage in bytes
    """
    import psutil
    proc = psutil.Process(os.getpid())
    #meminfo = proc.get_memory_info()
    meminfo = proc.memory_info()
    rss = meminfo[0]  # Resident Set Size / Mem Usage
    vms = meminfo[1]  # Virtual Memory Size / VM Size  # NOQA
    return rss


def get_matching_process_ids(cmd_pattern, user_pattern):
    """
    CommandLine:
        export PID=30196
        export PID=$(python -c "import utool as ut; print(ut.get_matching_process_ids('jonc', 'python2.7'))")
        export PID=$(python -c "import utool as ut; print(ut.get_matching_process_ids('jonc', 'matlab'))")
        sudo -H echo $PID
        ps -o pid,comm,nice -p $PID
        renice 10 -p $PID
        sudo renice -4 -p $PID

    user_pattern = 'jonc'
    cmd_pattern = 'main.py'
    user_pattern = None
    cmd_pattern = 'matlab'
    get_matching_process_ids(cmd_pattern, user_pattern)
    """
    import psutil
    import re
    process_list = list(psutil.process_iter())
    def matches_pattern(proc, user_pattern, cmd_pattern):
        matches_user = True if user_pattern is None else re.match(user_pattern, proc.username())
        cmdline_str = ' '.join(proc.cmdline())
        matches_name = True if cmd_pattern is None else re.search(cmd_pattern, cmdline_str)
        return matches_user and matches_name
    filtered_proc_list = [proc for proc in process_list if matches_pattern(proc, user_pattern, cmd_pattern)]

    for proc in filtered_proc_list:
        print(' | '.join([str(proc.username()), str(proc.nice()), str(proc), ' '.join(proc.cmdline())]))
        #print(proc.cmdline())
        #print(proc.pid)
        #print('---')

    important_process_list = [proc for proc in process_list if proc.nice() < -4]
    for proc in important_process_list:
        print(' -- '.join([str(proc.username()), str(proc.nice()), str(proc), ' '.join(proc.cmdline())]))

    #for proc in filtered_proc_list:
    #    print('---')
    #    print(proc)
    #    print(proc.cmdline())
    #    print(proc.nice())
    #    print(proc.pid)
    filtered_pid_list = [proc.pid for proc in filtered_proc_list]
    return filtered_pid_list


def num_cpus():
    import psutil
    return psutil.cpu_count(logical=True)


def num_unused_cpus(thresh=10):
    """
    Returns the number of cpus with utilization less than `thresh` percent
    """
    import psutil
    cpu_usage = psutil.cpu_percent(percpu=True)
    return sum([p < thresh for p in cpu_usage])


def available_memory():
    """
    Returns total system wide available memory in bytes
    """
    import psutil
    return psutil.virtual_memory().available


def total_memory():
    """
    Returns total system wide memory in bytes
    """
    import psutil
    return psutil.virtual_memory().total


def used_memory():
    """
    Returns total system wide used memory in bytes
    """
    return total_memory() - available_memory()


def memstats():
    print(get_memstats_str())


def get_memstats_str():
    return '\n'.join([
        ('[psutil] total     = %s' % util_str.byte_str2(total_memory())),
        ('[psutil] available = %s' % util_str.byte_str2(available_memory())),
        ('[psutil] used      = %s' % util_str.byte_str2(used_memory())),
        ('[psutil] current   = %s' % util_str.byte_str2(current_memory_usage())),
    ])


def get_python_datastructure_sizes():
    """
    References:
        http://stackoverflow.com/questions/1331471/in-memory-size-of-python-stucture

    CommandLine:
        python -m utool.util_resources --test-get_python_datastructure_sizes

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_resources import *  # NOQA
        >>> import utool as ut  # NOQA
        >>> type_sizes = get_python_datastructure_sizes()
        >>> result = ut.repr4(type_sizes, sorted_=True)
        >>> print(result)
    """
    import sys
    import decimal
    import six

    empty_types = {
        'int'     : 0,
        'float'   : 0.0,
        'dict'    : dict(),
        'set'     : set(),
        'tuple'   : tuple(),
        'list'    : list(),
        'str'     : '',
        'unicode' : u'',
        'decimal' : decimal.Decimal(0),
        'object'  : object(),
    }
    type_sizes = {key: sys.getsizeof(val)
                  for key, val in six.iteritems(empty_types)}
    return type_sizes

#psutil.virtual_memory()
#psutil.swap_memory()
#psutil.disk_partitions()
#psutil.disk_usage("/")
#psutil.disk_io_counters()
#psutil.net_io_counters(pernic=True)
#psutil.get_users()
#psutil.get_boot_time()
#psutil.get_pid_list()

if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_resources
        python -m utool.util_resources --allexamples
        python -m utool.util_resources --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
