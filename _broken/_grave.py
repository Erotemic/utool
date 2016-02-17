def ipython_execstr2():
    return textwrap.dedent(r'''
    import sys
    embedded = False
    try:
        __IPYTHON__
        in_ipython = True
    except NameError:
        in_ipython = False
    try:
        import IPython
        have_ipython = True
    except NameError:
        have_ipython = False
    if in_ipython:
        print('Presenting in current ipython shell.')
    elif '--cmd' in sys.argv:
        print('[utool.dbg] Requested IPython shell with --cmd argument.')
        if have_ipython:
            print('[utool.dbg] Found IPython')
            try:
                import IPython
                print('[utool.dbg] Presenting in new ipython shell.')
                embedded = True
                IPython.embed()
            except Exception as ex:
                print(repr(ex)+'\n!!!!!!!!')
                embedded = False
        else:
            print('[utool.dbg] IPython is not installed')
    ''')

# --- Exec Strings ---
IPYTHON_EMBED_STR = r'''
try:
    import IPython
    print('Presenting in new ipython shell.')
    embedded = True
    IPython.embed()
except Exception as ex:
    warnings.warn(repr(ex)+'\n!!!!!!!!')
    embedded = False
'''



def roundrobin(*iterables):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    raise NotImplementedError('not sure if this implementation is correct')
    # http://stackoverflow.com/questions/11125212/interleaving-lists-in-python
    #sentinel = object()
    #return (x for x in chain(*zip_longest(fillvalue=sentinel, *iterables)) if x is not sentinel)
    pending = len(iterables)
    if six.PY2:
        nexts = cycle(iter(it).next for it in iterables)
    else:
        nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))



def interleave2(*iterables):
    #from six.moves import izip_longest
    #izip_longest(args)
    raise NotImplementedError('not sure if this implementation is correct')
    return chain.from_iterable(zip(*iterables))


def interleave3(*args):
    cycle_iter = zip(*args)
    raise NotImplementedError('not sure if this implementation is correct')
    if six.PY2:
        for iter_ in cycle_iter:
            yield iter_.next()
    else:
        for iter_ in cycle_iter:
            yield next(iter_)


def commas(num, n=8):
    if util_type.is_float(num):
        #ret = sigfig_str(num, n=2)
        ret = '%.3f' % num
        return ret
        #return fewest_digits_float_str(num, n)
    return '%d' % num
    #return int_comma_str(num)


def fewest_digits_float_str(num, n=8):
    int_part = int(num)
    dec_part = num - int_part
    x = decimal.Decimal(dec_part, decimal.Context(prec=8))
    decimal_list = x.as_tuple()[1]
    nonzero_pos = 0
    for i in range(0, min(len(decimal_list), n)):
        if decimal_list[i] != 0:
            nonzero_pos = i
    sig_dec = int(dec_part * 10 ** (nonzero_pos + 1))
    float_str = int_comma_str(int_part) + '.' + str(sig_dec)
    return float_str
    #x.as_tuple()[n]


def format_(num, n=8):
    """
        makes numbers pretty e.g.
        nums = [9001, 9.053]
        print([format_(num) for num in nums])
    """
    if num is None:
        return 'None'
    if util_type.is_float(num):
        ret = ('%.' + str(n) + 'E') % num
        exp_pos  = ret.find('E')
        exp_part = ret[(exp_pos + 1):]
        exp_part = exp_part.replace('+', '')
        if exp_part.find('-') == 0:
            exp_part = '-' + exp_part[1:].strip('0')
        exp_part = exp_part.strip('0')
        if len(exp_part) > 0:
            exp_part = 'E' + exp_part
        flt_part = ret[:exp_pos].strip('0').strip('.')
        ret = flt_part + exp_part
        return ret
    return '%d' % num
