# -*- coding: utf-8 -*-
"""
Hashing convinience functions

You should opt to use a hash*27 function over a hash* function.


TODO: the same hashing algorithm should be used everywhere
Currently there is a mix of sha1, sha256, and sha512 in different places.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import hashlib
import copy
import os
import six
import uuid
import random
import warnings
from six.moves import zip, map
from utool import util_inject
from utool import util_path
from utool import util_type
(print, rrr, profile) = util_inject.inject2(__name__, '[hash]')

if util_type.HAVE_NUMPY:
    import numpy as np

# default length of hash codes
HASH_LEN = 16
HASH_LEN2 = 32

# HEX alphabet
ALPHABET_16 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f']

# A large base-54 alphabet (all chars are valid for filenames but not pretty)
ALPHABET_54 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z', ';', '=', '@', '[',
               ']', '^', '_', '`', '{', '}', '~', '!', '#', '$',
               '%', '&', '+', ',']


# A large base-41 alphabet (prettier subset of base 54)
ALPHABET_41 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z', '@', '!', '%', '&',
               '+']

ALPHABET_27 = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# ALPHABET = ALPHABET_41
ALPHABET = ALPHABET_27
BIGBASE = len(ALPHABET)


DictProxyType = type(object.__dict__)


def make_hash(o):
    r"""
    Makes a hash from a dictionary, list, tuple or set to any level, that
    contains only other hashable types (including any lists, tuples, sets, and
    dictionaries). In the case where other kinds of objects (like classes) need
    to be hashed, pass in a collection of object attributes that are pertinent.
    For example, a class can be hashed in this fashion:

    make_hash([cls.__dict__, cls.__name__])

    A function can be hashed like so:

    make_hash([fn.__dict__, fn.__code__])

    References:
        http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
    """
    if type(o) == DictProxyType:
        o2 = {}
        for k, v in o.items():
            if not k.startswith("__"):
                o2[k] = v
            o = o2
    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        return hash(o)
    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)
    return hash(tuple(frozenset(sorted(new_o.items()))))


def hashstr_arr27(arr, lbl, alphabet=ALPHABET_27, **kwargs):
    return hashstr_arr(arr, lbl, alphabet=alphabet, **kwargs)


def hashstr27(data, alphabet=ALPHABET_27, **kwargs):
    return hashstr(data, alphabet=alphabet, **kwargs)


def hashstr_arr(arr, lbl='arr', pathsafe=False, **kwargs):
    r"""
    Args:
        arr (ndarray):
        lbl (str): (default = 'arr')
        pathsafe (bool): (default = False)

    Returns:
        str: arr_hashstr

    CommandLine:
        python -m utool.util_hash --test-hashstr_arr
        python -m utool.util_hash hashstr_arr:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        >>> lbl = 'arr'
        >>> kwargs = {}
        >>> pathsafe = False
        >>> arr_hashstr = hashstr_arr(arr, lbl, pathsafe, alphabet=ALPHABET_27)
        >>> result = ('arr_hashstr = %s' % (str(arr_hashstr),))
        >>> print(result)
        arr_hashstr = arr((2,3)daukyreqnhfejkfs)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import numpy as np
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        >>> kwargs = {}
        >>> lbl = 'arr'
        >>> pathsafe = True
        >>> arr_hashstr = hashstr_arr(arr, lbl, pathsafe, alphabet=ALPHABET_27)
        >>> result = ('arr_hashstr = %s' % (str(arr_hashstr),))
        >>> print(result)
        arr_hashstr = arr-_2,3_daukyreqnhfejkfs-
    """
    if isinstance(arr, list):
        arr = tuple(arr)  # force arrays into a tuple for hashability
        # TODO: maybe for into numpy array instead? tuples might have problems
    if pathsafe:
        lbrace1, rbrace1, lbrace2, rbrace2 = '_', '_', '-', '-'
    else:
        lbrace1, rbrace1, lbrace2, rbrace2 = '(', ')', '(', ')'
    if isinstance(arr, tuple):
        arr_shape = lbrace1 + str(len(arr)) + rbrace1
    else:
        # Arr should be an ndarray here. append info about the ndarray
        arr_shape = lbrace1 + ','.join(list(map(str, arr.shape))) + rbrace1
    arr_hashstr_ = hashstr(arr, **kwargs)
    arr_hashstr = ''.join([lbl, lbrace2, arr_shape, arr_hashstr_, rbrace2])
    return arr_hashstr


def hashid_arr(arr, label='arr', hashlen=16):
    """ newer version of hashstr_arr2 """
    hashstr = hash_data(arr)[0:hashlen]
    if isinstance(arr, (list, tuple)):
        shapestr = len(arr)
    else:
        shapestr = ','.join(list(map(str, arr.shape)))
    hashid = '{}-{}-{}'.format(label, shapestr, hashstr)
    return hashid

if six.PY2:
    stringlike = (basestring, bytes)  # NOQA
if six.PY3:
    stringlike = (str, bytes)  # NOQA


def _covert_to_hashable(data):
    r"""
    Args:
        data (?):

    Returns:
        ?:

    CommandLine:
        python -m utool.util_hash _covert_to_hashable

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> from utool.util_hash import _covert_to_hashable  # NOQA
        >>> import utool as ut
        >>> data = np.array([1], dtype=np.int64)
        >>> result = _covert_to_hashable(data)
        >>> print(result)
    """
    if isinstance(data, six.binary_type):
        hashable = data
        prefix = b'TXT'
    elif util_type.HAVE_NUMPY and isinstance(data, np.ndarray):
        if data.dtype.kind == 'O':
            msg = '[ut] hashing ndarrays with dtype=object is unstable'
            warnings.warn(msg, RuntimeWarning)
            hashable = data.dumps()
        else:
            hashable = data.tobytes()
        prefix = b'NDARR'
    elif isinstance(data, six.text_type):
        # convert unicode into bytes
        hashable = data.encode('utf-8')
        prefix = b'TXT'
    elif isinstance(data, uuid.UUID):
        hashable = data.bytes
        prefix = b'UUID'
    elif isinstance(data, int):
        # warnings.warn('[util_hash] Hashing ints is slow, numpy is prefered')
        hashable = _int_to_bytes(data)
        # hashable = data.to_bytes(8, byteorder='big')
        prefix = b'INT'
    # elif isinstance(data, float):
    #     hashable = repr(data).encode('utf8')
    #     prefix = b'FLT'
    elif util_type.HAVE_NUMPY and isinstance(data, np.int64):
        return _covert_to_hashable(int(data))
    elif util_type.HAVE_NUMPY and isinstance(data, np.float64):
        a, b = float(data).as_integer_ratio()
        hashable = (a.to_bytes(8, byteorder='big') +
                    b.to_bytes(8, byteorder='big'))
        prefix = b'FLOAT'
    else:
        raise TypeError('unknown hashable type=%r' % (type(data)))
        # import bencode
        # hashable = bencode.Bencoder.encode(data).encode('utf-8')
        # prefix = b'BEN'
    prefix = b''
    return prefix, hashable


def _update_hasher(hasher, data):
    """
    This is the clear winner over the generate version.
    Used by hash_data

    Ignore:
        import utool
        rng = np.random.RandomState(0)
        # str1 = rng.rand(0).dumps()
        str1 = b'SEP'
        str2 = rng.rand(10000).dumps()
        for timer in utool.Timerit(100, label='twocall'):
            hasher = hashlib.sha256()
            with timer:
                hasher.update(str1)
                hasher.update(str2)
        a = hasher.hexdigest()
        for timer in utool.Timerit(100, label='concat'):
            hasher = hashlib.sha256()
            with timer:
                hasher.update(str1 + str2)
        b = hasher.hexdigest()
        assert a == b
        # CONCLUSION: Faster to concat in case of prefixes and seps

        nested_data = {'1': [rng.rand(100), '2', '3'],
                       '2': ['1', '2', '3', '4', '5'],
                       '3': [('1', '2'), ('3', '4'), ('5', '6')]}
        data = list(nested_data.values())


        for timer in utool.Timerit(1000, label='cat-generate'):
            hasher = hashlib.sha256()
            with timer:
                hasher.update(b''.join(_bytes_generator(data)))

        for timer in utool.Timerit(1000, label='inc-generate'):
            hasher = hashlib.sha256()
            with timer:
                for b in _bytes_generator(data):
                    hasher.update(b)

        for timer in utool.Timerit(1000, label='inc-generate'):
            hasher = hashlib.sha256()
            with timer:
                for b in _bytes_generator(data):
                    hasher.update(b)

        for timer in utool.Timerit(1000, label='chunk-inc-generate'):
            hasher = hashlib.sha256()
            import ubelt as ub
            with timer:
                for chunk in ub.chunks(_bytes_generator(data), 5):
                    hasher.update(b''.join(chunk))

        for timer in utool.Timerit(1000, label='inc-update'):
            hasher = hashlib.sha256()
            with timer:
                _update_hasher(hasher, data)

        data = ut.lorium_ipsum()
        hash_data(data)
        ut.hashstr27(data)
        %timeit hash_data(data)
        %timeit ut.hashstr27(repr(data))

        for timer in utool.Timerit(100, label='twocall'):
            hasher = hashlib.sha256()
            with timer:
                hash_data(data)

        hasher = hashlib.sha256()
        hasher.update(memoryview(np.array([1])))
        print(hasher.hexdigest())

        hasher = hashlib.sha256()
        hasher.update(np.array(['1'], dtype=object))
        print(hasher.hexdigest())

    """
    if isinstance(data, (tuple, list, zip)):
        needs_iteration = True
    elif (util_type.HAVE_NUMPY and isinstance(data, np.ndarray) and
          data.dtype.kind == 'O'):
        # ndarrays of objects cannot be hashed directly.
        needs_iteration = True
    else:
        needs_iteration = False

    if needs_iteration:
        # try to nest quickly without recursive calls
        SEP = b'SEP'
        iter_prefix = b'ITER'
        # if isinstance(data, tuple):
        #     iter_prefix = b'TUP'
        # else:
        #     iter_prefix = b'LIST'
        iter_ = iter(data)
        hasher.update(iter_prefix)
        try:
            for item in iter_:
                prefix, hashable = _covert_to_hashable(data)
                binary_data = SEP + prefix + hashable
                # b''.join([SEP, prefix, hashable])
                hasher.update(binary_data)
        except TypeError:
            # need to use recursive calls
            # Update based on current item
            _update_hasher(hasher, item)
            for item in iter_:
                # Ensure the items have a spacer between them
                hasher.update(SEP)
                _update_hasher(hasher, item)
    else:
        prefix, hashable = _covert_to_hashable(data)
        binary_data = prefix + hashable
        # b''.join([prefix, hashable])
        hasher.update(binary_data)


# def _bytes_generator(data):
#     # SLOWER METHOD
#     if isinstance(data, (tuple, list)):
#         # Ensure there is a iterable prefix with a spacer item
#         SEP = b'SEP'
#         iter_prefix = b'ITER'
#         # if isinstance(data, tuple):
#         #     iter_prefix = b'TUP'
#         # else:
#         #     iter_prefix = b'LIST'
#         iter_ = iter(data)
#         yield iter_prefix
#         try:
#             # try to nest quickly without recursive calls
#             for item in iter_:
#                 prefix, hashable = _covert_to_hashable(data)
#                 yield SEP
#                 yield prefix
#                 yield hashable
#         except TypeError:
#             # recover from failed item and then continue iterating using slow
#             # recursive calls
#             yield SEP
#             for bytes_ in _bytes_generator(item):
#                 yield bytes_
#             for item in iter_:
#                 yield SEP
#                 for bytes_ in _bytes_generator(item):
#                     yield bytes_
#     else:
#         prefix, hashable = _covert_to_hashable(data)
#         yield prefix
#         yield hashable


def b(x):
    return six.binary_type(six.b(x))


SEP_STR = '-'
SEP_BYTE = b(SEP_STR)


def freeze_hash_bytes(bytes_):
    import codecs
    hexstr = codecs.encode(bytes_, 'hex').decode('utf8')
    return hexstr


def combine_hashes(bytes_list, hasher=None):
    """
    Only works on bytes

    Example:
        >>> # DISABLE_DOCTEST
        >>> x = [b('1111'), b('2222')]
        >>> y = [b('11'), b('11'), b('22'), b('22')]
        >>> bytes_list = y
        >>> out1 = ut.combine_hashes(x, hashlib.sha1())
        >>> hasher = hashlib.sha1()
        >>> out2 = ut.combine_hashes(y, hasher)
        >>> bytes_ = out2
        >>> assert hasher.hexdigest() == freeze_hash_bytes(hasher.digest())
        >>> assert convert_bytes_to_bigbase(hasher.digest()) == convert_hexstr_to_bigbase(hasher.hexdigest())
        >>> assert out1 != out2
        >>> print('out1 = %r' % (out1,))
        >>> print('out2 = %r' % (out2,))
    """
    if hasher is None:
        hasher = hashlib.sha256()
    for b in bytes_list:
        hasher.update(b)
        hasher.update(SEP_BYTE)
    return hasher.digest()


@profile
def hash_data(data, hashlen=None, alphabet=None):
    r"""
    Get a unique hash depending on the state of the data.

    Args:
        data (object): any sort of loosely organized data
        hashlen (None): (default = None)
        alphabet (None): (default = None)

    Returns:
        str: text -  hash string

    CommandLine:
        python -m utool.util_hash hash_data

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import utool as ut
        >>> counter = [0]
        >>> failed = []
        >>> def check_hash(input_, want=None):
        >>>     count = counter[0] = counter[0] + 1
        >>>     got = ut.hash_data(input_)
        >>>     print('({}) {}'.format(count, got))
        >>>     if want is not None and not got.startswith(want):
        >>>         failed.append((got, input_, count, want))
        >>> check_hash('1', 'wuvrng')
        >>> check_hash(['1'], 'dekbfpby')
        >>> check_hash(tuple(['1']), 'dekbfpby')
        >>> check_hash(b'12', 'marreflbv')
        >>> check_hash([b'1', b'2'], 'nwfs')
        >>> check_hash(['1', '2', '3'], 'arfrp')
        >>> check_hash(['1', np.array([1,2,3]), '3'], 'uyqwcq')
        >>> check_hash('123', 'ehkgxk')
        >>> check_hash(zip([1, 2, 3], [4, 5, 6]), 'mjcpwa')
        >>> import numpy as np
        >>> rng = np.random.RandomState(0)
        >>> check_hash(rng.rand(100000), 'bdwosuey')
        >>> for got, input_, count, want in failed:
        >>>     print('failed {} on {}'.format(count, input_))
        >>>     print('got={}, want={}'.format(got, want))
        >>> assert not failed
    """
    if alphabet is None:
        alphabet = ALPHABET_27
    if hashlen is None:
        hashlen = HASH_LEN2
    if isinstance(data, stringlike) and len(data) == 0:
        # Make a special hash for empty data
        text = (alphabet[0] * hashlen)
    else:
        hasher = hashlib.sha512()
        _update_hasher(hasher, data)
        # Get a 128 character hex string
        text = hasher.hexdigest()
        # Shorten length of string (by increasing base)
        hashstr2 = convert_hexstr_to_bigbase(text, alphabet, bigbase=len(alphabet))
        # Truncate
        text = hashstr2[:hashlen]
        return text


def digest_data(data, alg='sha256'):
    hasher = {
        'md5'    : hashlib.md5,
        'sha1'   : hashlib.sha1,
        'sha256' : hashlib.sha256,
    }[alg]()
    _update_hasher(hasher, data)
    return hasher.digest()


def hashstr(data, hashlen=HASH_LEN, alphabet=ALPHABET):
    """
    python -c "import utool as ut; print(ut.hashstr('abcd'))"

    Args:
        data (hashable):
        hashlen (int): (default = 16)
        alphabet (list): list of characters:

    Returns:
        str: hashstr

    CommandLine:
        python -m utool.util_hash --test-hashstr
        python3 -m utool.util_hash --test-hashstr
        python3 -m utool.util_hash --test-hashstr:2
        python -m utool.util_hash hashstr:3
        python3 -m utool.util_hash hashstr:3


    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> data = 'foobar'
        >>> hashlen = 16
        >>> alphabet = ALPHABET_41
        >>> text = hashstr(data, hashlen, alphabet)
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)
        text = mi5yum60mbxhyp+x

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> data = ''
        >>> hashlen = 16
        >>> alphabet = ALPHABET_41
        >>> text = hashstr(data, hashlen, alphabet)
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)
        text = 0000000000000000

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import numpy as np
        >>> data = np.array([1, 2, 3])
        >>> hashlen = 16
        >>> alphabet = ALPHABET_41
        >>> text = hashstr(data, hashlen, alphabet)
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)
        text = z5lqw0bzt4dmb9yy

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import numpy as np
        >>> from uuid import UUID
        >>> data = (UUID('7cd0197b-1394-9d16-b1eb-0d8d7a60aedc'), UUID('c76b54a5-adb6-7f16-f0fb-190ab99409f8'))
        >>> hashlen = 16
        >>> alphabet = ALPHABET_41
        >>> text = hashstr_arr(data, 'label')
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)


    Ignore:
        >>> # DISABLE_DOCTEST
        >>> # UNSTABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import numpy as np
        >>> data = np.array(['a', 'b'], dtype=object)
        >>> text = hashstr(data, alphabet=ALPHABET_27)
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)

    Ignore:
        data = np.array(['a', 'b'], dtype=object)
        data.tobytes()
        data = np.array(['a', 'b'])
        data = ['a', 'b']
        data = np.array([1, 2, 3])
        import hashlib
        from six.moves import cPickle as pickle
        pickle.dumps(data, protocol=2)

        python2 -c "import hashlib, numpy; print(hashlib.sha1('ab').hexdigest())"
        python3 -c "import hashlib, numpy; print(hashlib.sha1('ab'.encode('utf8')).hexdigest())"

        python2 -c "import hashlib, numpy; print(hashlib.sha1('ab').hexdigest())"
        python3 -c "import hashlib, numpy; print(hashlib.sha1(b'ab').hexdigest())"

        python2 -c "import hashlib, numpy; print(hashlib.sha1(numpy.array([1, 2])).hexdigest())"
        python3 -c "import hashlib, numpy; print(hashlib.sha1(numpy.array([1, 2])).hexdigest())"

        # TODO: numpy arrays of strings must be encoded to bytes first in python3
        python2 -c "import hashlib, numpy; print(hashlib.sha1(numpy.array(['a', 'b'])).hexdigest())"
        python3 -c "import hashlib, numpy; print(hashlib.sha1(numpy.array([b'a', b'b'])).hexdigest())"

        python -c "import hashlib, numpy; print(hashlib.sha1(numpy.array(['a', 'b'], dtype=object)).hexdigest())"
        python -c "import hashlib, numpy; print(hashlib.sha1(numpy.array(['a', 'b'], dtype=object)).hexdigest())"
    """
    if util_type.HAVE_NUMPY and isinstance(data, np.ndarray):
        if data.dtype.kind == 'O':
            msg = '[ut] hashing ndarrays with dtype=object is unstable'
            warnings.warn(msg, RuntimeWarning)
            # but tobytes is ok, but differs between python 2 and 3 for objects
            data = data.dumps()
            # data = data.tobytes()
    if isinstance(data, tuple):
        # should instead do
        if False:
            hasher = hashlib.sha512()
            items = data
            for item in items:
                if isinstance(item, uuid.UUID):
                    hasher.update(item.bytes)
                else:
                    hasher.update(item)
            text = hasher.hexdigest()
            hashstr2 = convert_hexstr_to_bigbase(text, alphabet, bigbase=len(alphabet))
            # Truncate
            text = hashstr2[:hashlen]
            return text
        else:
            msg = '[ut] hashing tuples with repr is not a good idea. FIXME'
            # warnings.warn(msg, RuntimeWarning)
            data = repr(data)  # Hack?

    # convert unicode into raw bytes
    if isinstance(data, six.text_type):
        data = data.encode('utf-8')

    if isinstance(data, stringlike) and len(data) == 0:
        # Make a special hash for empty data
        text = (alphabet[0] * hashlen)
    else:
        # Get a 128 character hex string
        text = hashlib.sha512(data).hexdigest()
        # Shorten length of string (by increasing base)
        hashstr2 = convert_hexstr_to_bigbase(text, alphabet, bigbase=len(alphabet))
        # Truncate
        text = hashstr2[:hashlen]
    return text

r"""
def valid_filename_ascii_chars():
    # Find invalid chars
    ntfs_inval = '< > : " / \ | ? *'.split(' ')
    other_inval = [' ', '\'', '.']
    #case_inval = map(chr, range(97, 123))
    case_inval = map(chr, range(65, 91))
    invalid_chars = set(ntfs_inval + other_inval + case_inval)
    # Find valid chars
    valid_chars = []
    for index in range(32, 127):
        char = chr(index)
        if not char in invalid_chars:
            print index, chr(index)
            valid_chars.append(chr(index))
    return valid_chars
valid_filename_ascii_chars()
"""

if six.PY3:
    def _int_to_bytes(int_):
        length = max(4, int_.bit_length())
        bytes_ = int_.to_bytes(length, byteorder='big')
        # bytes_ = int_.to_bytes(4, byteorder='big')
        # int_.to_bytes(8, byteorder='big')  # TODO: uncomment
        return bytes_

    def _bytes_to_int(bytes_):
        int_ = int.from_bytes(bytes_, 'big')
        return int_
else:
    def _py2_to_bytes(int_, length, byteorder='big'):
        h = '%x' % int_
        s = ('0' * (len(h) % 2) + h).zfill(length * 2).decode('hex')
        bytes_ =  s if byteorder == 'big' else s[::-1]
        return bytes_

    import codecs
    def _int_to_bytes(int_):
        length = max(4, int_.bit_length())
        bytes_ = _py2_to_bytes(int_, length, 'big')
        # bytes_ = struct.pack('>i', int_)
        return bytes_

    def _bytes_to_int(bytes_):
        int_ = int(codecs.encode(bytes_, 'hex'), 16)
        # int_ = struct.unpack('>i', bytes_)[0]
        # int_ = struct.unpack_from('>L', bytes_)[0]
        return int_


def _test_int_byte_conversion():
    import itertools as it
    import utool as ut
    inputs = list(it.chain(
        range(0, 10),
        (2 ** i for i in range(0, 256, 32)),
        (2 ** i + 1 for i in range(0, 256, 32)),
    ))
    for int_0 in inputs:
        print('---')
        print('int_0 = %s' % (ut.repr2(int_0),))
        bytes_ = _int_to_bytes(int_0)
        int_ = _bytes_to_int(bytes_)
        print('bytes_ = %s' % (ut.repr2(bytes_),))
        print('int_ = %s' % (ut.repr2(int_),))
        assert int_ == int_0


def convert_bytes_to_bigbase(bytes_, alphabet=ALPHABET_27):
    r"""
    Args:
        bytes_ (bytes):

    Returns:
        str:

    Ignore:

    CommandLine:
        python -m utool.util_hash convert_bytes_to_bigbase

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import utool as ut
        >>> bytes_ = b('9999999999999999999999999999999999')
        >>> alphabet = ALPHABET_27
        >>> result = convert_bytes_to_bigbase(bytes_, alphabet)
        >>> print(result)
        fervudwhpustklnptklklcgswbmvtustqocdpgiwkgrvwytvneardkpytd
    """
    x = _bytes_to_int(bytes_)
    if x == 0:
        return '0'
    sign = 1 if x > 0 else -1
    x *= sign
    digits = []
    bigbase = len(alphabet)
    while x:
        digits.append(alphabet[x % bigbase])
        x //= bigbase
    if sign < 0:
        digits.append('-')
        digits.reverse()
    newbase_str = ''.join(digits)
    return newbase_str


def convert_hexstr_to_bigbase(hexstr, alphabet=ALPHABET, bigbase=BIGBASE):
    r"""
    Packs a long hexstr into a shorter length string with a larger base

    Ignore:
        # Determine the length savings with lossless conversion
        import sympy as sy
        consts = dict(hexbase=16, hexlen=256, bigbase=27)
        symbols = sy.symbols('hexbase, hexlen, bigbase, newlen')
        haexbase, hexlen, bigbase, newlen = symbols
        eqn = sy.Eq(16 ** hexlen,  bigbase ** newlen)
        newlen_ans = sy.solve(eqn, newlen)[0].subs(consts).evalf()
        print('newlen_ans = %r' % (newlen_ans,))

        # for a 27 char alphabet we can get 216
        print('Required length for lossless conversion len2 = %r' % (len2,))

        def info(base, len):
            bits = base ** len
            print('base = %r' % (base,))
            print('len = %r' % (len,))
            print('bits = %r' % (bits,))
        info(16, 256)
        info(27, 16)
        info(27, 64)
        info(27, 216)
    """
    x = int(hexstr, 16)  # first convert to base 16
    if x == 0:
        return '0'
    sign = 1 if x > 0 else -1
    x *= sign
    digits = []
    while x:
        digits.append(alphabet[x % bigbase])
        x //= bigbase
    if sign < 0:
        digits.append('-')
        digits.reverse()
    newbase_str = ''.join(digits)
    return newbase_str


def hashstr_md5(data):
    """
    Ignore:
        >>> # xdoctest: +SKIP
        >>> from utool.util_hash import *  # NOQA
        >>> fpath = ut.grab_test_imgpath('patsy.jpg')
        %timeit ut.get_file_hash(fpath, hasher=hashlib.sha1())
        %timeit ut.get_file_hash(fpath, hasher=hashlib.md5())

    """
    text = hashlib.md5(data).hexdigest()
    #bin(int(my_hexdata, scale))
    return text


def hashstr_sha1(data, base10=False):
    text = hashlib.sha1(data).hexdigest()
    if base10:
        text = int("0x" + text, 0)
    return text


def get_file_hash(fpath, blocksize=65536, hasher=None, stride=1,
                  hexdigest=False):
    r"""
    For better hashes use hasher=hashlib.sha256, and keep stride=1

    Args:
        fpath (str):  file path string
        blocksize (int): 2 ** 16. Affects speed of reading file
        hasher (None):  defaults to sha1 for fast (but insecure) hashing
        stride (int): strides > 1 skip data to hash, useful for faster
                      hashing, but less accurate, also makes hash dependant on
                      blocksize.

    References:
        http://stackoverflow.com/questions/3431825/generating-a-md5-checksum-of-a-file
        http://stackoverflow.com/questions/5001893/when-should-i-use-sha-1-and-when-should-i-use-sha-2

    CommandLine:
        python -m utool.util_hash --test-get_file_hash
        python -m utool.util_hash --test-get_file_hash:0
        python -m utool.util_hash --test-get_file_hash:1

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> fpath = ut.grab_test_imgpath('patsy.jpg')
        >>> #blocksize = 65536  # 2 ** 16
        >>> blocksize = 2 ** 16
        >>> hasher = None
        >>> stride = 1
        >>> hashbytes_20 = get_file_hash(fpath, blocksize, hasher, stride)
        >>> result = repr(hashbytes_20)
        >>> print(result)
        '7\x07B\x0eX<sRu\xa2\x90P\xda\xb2\x84?\x81?\xa9\xd9'

        '\x13\x9b\xf6\x0f\xa3QQ \xd7"$\xe9m\x05\x9e\x81\xf6\xf2v\xe4'

        '\x16\x00\x80Xx\x8c-H\xcdP\xf6\x02\x9frl\xbf\x99VQ\xb5'

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> #fpath = ut.grab_file_url('http://en.wikipedia.org/wiki/List_of_comets_by_type')
        >>> fpath = ut.unixjoin(ut.ensure_app_resource_dir('utool'), 'tmp.txt')
        >>> ut.write_to(fpath, ut.lorium_ipsum())
        >>> blocksize = 2 ** 3
        >>> hasher = None
        >>> stride = 2
        >>> hashbytes_20 = get_file_hash(fpath, blocksize, hasher, stride)
        >>> result = repr(hashbytes_20)
        >>> print(result)
        '5KP\xcf>R\xf6\xffO:L\xac\x9c\xd3V+\x0e\xf6\xe1n'

    Ignore:
        file_ = open(fpath, 'rb')
    """
    if hasher is None:
        hasher = hashlib.sha1()
    with open(fpath, 'rb') as file_:
        buf = file_.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            if stride > 1:
                file_.seek(blocksize * (stride - 1), 1)  # skip blocks
            buf = file_.read(blocksize)
        if hexdigest:
            return hasher.hexdigest()
        else:
            return hasher.digest()


def write_hash_file(fpath, hash_tag='md5', recompute=False):
    r""" Creates a hash file for each file in a path

    CommandLine:
        python -m utool.util_hash --test-write_hash_file

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> import utool as ut
        >>> from utool.util_hash import *  # NOQA
        >>> fpath = ut.grab_test_imgpath('patsy.jpg')
        >>> write_hash_file(fpath, 'md5')
    """
    hash_dict = {
        'md5'    : hashlib.md5(),
        'sha1'   : hashlib.sha1(),
        'sha256' : hashlib.sha256(),
    }
    message = "Unrecognized hashing function.  Use 'md5', 'sha1', or 'sha256"
    assert hash_tag in hash_dict, message
    if fpath.endswith('.%s' % (hash_tag, )):
        # No need to compute hashes on hashes
        return
    # Get hash path
    hash_fpath = '%s.%s' % (fpath, hash_tag, )
    if os.path.exists(hash_fpath) and not recompute:
        return
    # Assert this is a file
    file_type = util_path.get_path_type(fpath)
    if file_type == 'file':
        # Compute hash
        hasher = hash_dict[hash_tag]
        hash_local = get_file_hash(fpath, hasher=hasher, hexdigest=True)
        print('[utool] Adding:', fpath, hash_local)
        with open(hash_fpath, 'w') as hash_file:
            hash_file.write(hash_local)
        return hash_fpath


def write_hash_file_for_path(path, recompute=False):
    r""" Creates a hash file for each file in a path

    CommandLine:
        python -m utool.util_hash --test-write_hash_file_for_path

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> import os
        >>> import utool as ut
        >>> from utool.util_hash import *  # NOQA
        >>> fpath = ut.grab_test_imgpath('patsy.jpg')
        >>> path, _ = os.path.split(fpath)
        >>> hash_fpath_list = write_hash_file_for_path(path)
        >>> for hash_fpath in hash_fpath_list:
        >>>     assert os.path.exists(hash_fpath)
        >>>     ut.delete(hash_fpath)
    """
    hash_fpath_list = []
    for root, dname_list, fname_list in os.walk(path):
        for fname in sorted(fname_list):
            # fpath = os.path.join(path, fname)
            fpath = os.path.join(root, fname)
            hash_fpath = write_hash_file(fpath, recompute=recompute)
            if hash_fpath is not None:
                hash_fpath_list.append(hash_fpath)
    return hash_fpath_list


def get_file_uuid(fpath, hasher=None, stride=1):
    """ Creates a uuid from the hash of a file
    """
    if hasher is None:
        hasher = hashlib.sha1()  # 20 bytes of output
        #hasher = hashlib.sha256()  # 32 bytes of output
    # sha1 produces a 20 byte hash
    hashbytes_20 = get_file_hash(fpath, hasher=hasher, stride=stride)
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


def image_uuid(pil_img):
    """
    UNSAFE: DEPRICATE: JPEG IS NOT GAURENTEED TO PRODUCE CONSITENT VALUES ON

    MULTIPLE MACHINES image global unique id

    References:
        http://stackoverflow.com/questions/23565889/jpeg-images-have-different-pixel-values-across-multiple-devices
    """
    print('WARNING DO NOT USE utool.util_hash.image_uuid UNSAFE AND DEPRICATED')
    # Get the bytes of the image
    img_bytes_ = pil_img.tobytes()
    uuid_ = hashable_to_uuid(img_bytes_)
    return uuid_


def augment_uuid(uuid_, *hashables):
    #from six.moves import reprlib
    #uuidhex_data   = uuid_.get_bytes()
    uuidhex_data   = uuid_.bytes
    #hashable_str    = ''.join(map(repr, hashables))
    # Python 2 and 3 diverge here because repr returns
    # ascii data in python2 and unicode text in python3
    # it would be nice to
    # warnings.warn('[ut] should not use repr when hashing', RuntimeWarning)
    def tmprepr(x):
        y = repr(x)
        # hack to remove u prefix
        if isinstance(x, six.string_types):
            if y.startswith('u'):
                y = y[1:]
        return y
    if six.PY2:
        hashable_text = ''.join(map(tmprepr, hashables))
        hashable_data = hashable_text.encode('utf-8')
        #hashable_data = b''.join(map(bytes, hashables))
    elif six.PY3:
        hashable_text    = ''.join(map(tmprepr, hashables))
        hashable_data = hashable_text.encode('utf-8')
        #hashable_data = b''.join(map(bytes, hashables))
    augmented_data   = uuidhex_data + hashable_data
    augmented_uuid_ = hashable_to_uuid(augmented_data)
    return augmented_uuid_


@profile
def combine_uuids(uuids, ordered=True, salt=''):
    """
    Creates a uuid that specifies a group of UUIDS

    Args:
        uuids (list): list of uuid objects
        ordered (bool): if False uuid order changes the resulting combined uuid
            otherwise the uuids are considered an orderless set
        salt (str): salts the resulting hash

    Returns:
        uuid.UUID: combined uuid

    CommandLine:
        python -m utool.util_hash --test-combine_uuids

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import utool as ut
        >>> uuids = [hashable_to_uuid('one'), hashable_to_uuid('two'),
        >>>          hashable_to_uuid('three')]
        >>> combo1 = combine_uuids(uuids, ordered=True)
        >>> combo2 = combine_uuids(uuids[::-1], ordered=True)
        >>> combo3 = combine_uuids(uuids, ordered=False)
        >>> combo4 = combine_uuids(uuids[::-1], ordered=False)
        >>> result = ut.repr4([combo1, combo2, combo3, combo4], nobr=True)
        >>> print(result)
        UUID('83ee781f-8646-ccba-0ed8-13842825c12a'),
        UUID('52bbb33f-612e-2ab8-a62c-2f46e5b1edc8'),
        UUID('945cadab-e834-e581-0f74-62f106d20d81'),
        UUID('945cadab-e834-e581-0f74-62f106d20d81'),

    Example:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import utool as ut
        >>> uuids = [uuid.UUID('5ff6b34e-7d8f-ef32-5fad-489266acd2ae'),
        >>>          uuid.UUID('f2400146-ec12-950b-1489-668228e155a8'),
        >>>          uuid.UUID('037d6f31-8c73-f961-1fe4-d616442a1e86'),
        >>>          uuid.UUID('ca45d6e2-e648-09cc-a49e-e71c6fa3b3f3')]
        >>> ordered = True
        >>> salt = u''
        >>> result = combine_uuids(uuids, ordered, salt)
        >>> print(result)
        1dabc66b-b564-676a-99b4-5cae7a9e7294
    """
    if len(uuids) == 0:
        return get_zero_uuid()
    elif len(uuids) == 1:
        return uuids[0]
    else:
        if not ordered:
            uuids = sorted(uuids)
        sep_str = '-'
        sep_byte = six.binary_type(six.b(sep_str))
        pref = six.binary_type(six.b('{}{}{}'.format(salt, sep_str, len(uuids))))
        combined_bytes = pref + sep_byte.join([u.bytes for u in uuids])
        combined_uuid = hashable_to_uuid(combined_bytes)
        return combined_uuid


if six.PY3:
    def _ensure_hashable_bytes(hashable_):
        # If hashable_ is text (python3)
        if isinstance(hashable_, bytes):
            return hashable_
        elif isinstance(hashable_, str):
            return hashable_.encode('utf-8')
        elif isinstance(hashable_, int):
            return hashable_.to_bytes(4, byteorder='big')
            # return int_.to_bytes(8, byteorder='big')  # TODO: uncomment
        elif isinstance(hashable_, (list, tuple)):
            return str(hashable_).encode('utf-8')
        else:
            return hashable_
elif six.PY2:
    import struct
    def _ensure_hashable_bytes(hashable_):
        # If hashable_ is data (python2)
        if isinstance(hashable_, bytes):
            return hashable_
        elif isinstance(hashable_, str):
            return hashable_.encode('utf-8')
        elif isinstance(hashable_, int):
            return struct.pack('>i', hashable_)
        elif isinstance(hashable_, (list, tuple)):
            return str(hashable_).encode('utf-8')
        else:
            return bytes(hashable_)


def hashable_to_uuid(hashable_):
    """
    TODO: ensure that python2 and python3 agree on hashes of the same
    information

    Args:
        hashable_ (hashable): hashables are bytes-like objects
           An object that supports the Buffer Protocol, like bytes, bytearray
           or memoryview. Bytes-like objects can be used for various operations
           that expect binary data, such as compression, saving to a binary
           file or sending over a socket. Some operations need the binary data
           to be mutable, in which case not all bytes-like objects can apply.

    Returns:
        UUID: uuid_

    CommandLine:
        python -m utool.util_hash --test-hashable_to_uuid

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import utool as ut
        >>> hashables = [
        >>>     'foobar',
        >>>     'foobar'.encode('utf-8'),
        >>>     u'foobar',
        >>>     10,
        >>>     [1, 2, 3],
        >>> ]
        >>> uuids = []
        >>> for hashable_ in hashables:
        >>>     uuid_ = hashable_to_uuid(hashable_)
        >>>     uuids.append(uuid_)
        >>> result = ut.repr4(ut.lmap(str, uuids), strvals=True, nobr=True)
        >>> print(result)
        8843d7f9-2416-211d-e9eb-b963ff4ce281,
        8843d7f9-2416-211d-e9eb-b963ff4ce281,
        8843d7f9-2416-211d-e9eb-b963ff4ce281,
        e864ece8-8880-43b6-7277-c8b2cefe96ad,
        a01eda32-e4e0-b139-3274-e91d1b3e9ecf,
    """
    bytes_ = _ensure_hashable_bytes(hashable_)
    try:
        bytes_sha1 = hashlib.sha1(bytes_)
    except TypeError:
        print('hashable_ = %r' % (hashable_,))
        print('bytes_ = %r' % (bytes_,))
        raise
    # Digest them into a hash
    hashbytes_20 = bytes_sha1.digest()
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


def random_uuid():
    return uuid.uuid4()


def random_nonce(length=64, alphabet=None):
    """
    returns a random string of len=<length> from <alphabet>
    I have no idea why this is named random_nonce
    """
    assert length > 0
    if alphabet is None:
        alphabet = ALPHABET_16
    return ''.join( [alphabet[random.randint(0, len(alphabet) - 1)] for _ in range(length)] )


def get_zero_uuid():
    return uuid.UUID('00000000-0000-0000-0000-000000000000')

# Cleanup namespace
# del ALPHABET_41
del ALPHABET_54


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_hash
        python -m utool.util_hash --allexamples
        python3 -m utool.util_hash --allexamples
        python -m utool.util_hash --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
