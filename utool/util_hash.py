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
from utool import util_inject
from utool import util_path
from utool import util_type
(print, rrr, profile) = util_inject.inject2(__name__, '[hash]')

if util_type.HAVE_NUMPY:
    import numpy as np

# default length of hash codes
HASH_LEN = 16

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


ALPHABET = ALPHABET_41
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


if six.PY2:
    stringlike = (basestring, bytes)
if six.PY3:
    stringlike = (str, bytes)


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


    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> data = 'foobar'
        >>> hashlen = 16
        >>> alphabet = ALPHABET
        >>> text = hashstr(data, hashlen, alphabet)
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)
        text = mi5yum60mbxhyp+x

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> data = ''
        >>> hashlen = 16
        >>> alphabet = ALPHABET
        >>> text = hashstr(data, hashlen, alphabet)
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)
        text = 0000000000000000

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import numpy as np
        >>> data = np.array([1, 2, 3])
        >>> hashlen = 16
        >>> alphabet = ALPHABET
        >>> text = hashstr(data, hashlen, alphabet)
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)
        text = z5lqw0bzt4dmb9yy

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import numpy as np
        >>> from uuid import UUID
        >>> data = (UUID('7cd0197b-1394-9d16-b1eb-0d8d7a60aedc'), UUID('c76b54a5-adb6-7f16-f0fb-190ab99409f8'))
        >>> hashlen = 16
        >>> alphabet = ALPHABET
        >>> text = hashstr_arr(data, 'label')
        >>> result = ('text = %s' % (str(text),))
        >>> print(result)


    Example3:
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
                    hasher.update(item.bytes_)
                else:
                    hasher.update(item)
            text = hasher.hexdigest()
            hashstr2 = convert_hexstr_to_bigbase(text, alphabet, bigbase=len(alphabet))
            # Truncate
            text = hashstr2[:hashlen]
            return text
        else:
            msg = '[ut] hashing tuples with repr is not a good idea. FIXME'
            warnings.warn(msg, RuntimeWarning)
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

"""
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


def convert_hexstr_to_bigbase(hexstr, alphabet=ALPHABET, bigbase=BIGBASE):
    """ Packs a long hexstr into a shorter length string with a larger base
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
    >>> from utool.util_hash import *  # NOQA
    >>> fpath = ut.grab_test_imgpath('patsy.jpg')
    >>> %timeit ut.get_file_hash(fpath, hasher=hashlib.sha1())
    >>> %timeit ut.get_file_hash(fpath, hasher=hashlib.md5())

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

    Example:
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

    Example:
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


def write_hash_file(fpath, hash_tag='md5', recompute=False, **kwargs):
    r""" Creates a hash file for each file in a path

    CommandLine:
        python -m utool.util_hash --test-write_hash_file

    Example:
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


def write_hash_file_for_path(path, **kwargs):
    r""" Creates a hash file for each file in a path

    CommandLine:
        python -m utool.util_hash --test-write_hash_file_for_path

    Example:
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
            fpath = os.path.join(path, fname)
            hash_fpath = write_hash_file(fpath, **kwargs)
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
    warnings.warn('[ut] should not use repr when hashing', RuntimeWarning)
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


def combine_uuids(uuids, ordered=True, salt=''):
    """
    Creates a uuid that specifies a group of UUIDS

    Args:
        uuids (list): list of uuid objects
        ordered (bool): if False uuid order changes the resulting combined uuid
            otherwise the uuids are considered an orderless set

    Returns:
        uuid.UUID: combined uuid

    CommandLine:
        python -m utool.util_hash combine_uuids

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> import utool as ut
        >>> uuids = [hashable_to_uuid('one'), hashable_to_uuid('two'), hashable_to_uuid('three')]
        >>> combo1 = combine_uuids(uuids, ordered=True)
        >>> combo2 = combine_uuids(uuids[::-1], ordered=True)
        >>> combo3 = combine_uuids(uuids, ordered=False)
        >>> combo4 = combine_uuids(uuids[::-1], ordered=False)
        >>> result = ut.repr4([combo1, combo2, combo3, combo4])
        >>> print(result)
        [
            UUID('ce60fdb8-6b94-c77b-108c-1ff9579dde0d'),
            UUID('57f6e6c8-7ddc-162d-42bf-29013fe399ed'),
            UUID('2b8c46b7-8d0d-23c6-a81e-3eb453b2eb04'),
            UUID('2b8c46b7-8d0d-23c6-a81e-3eb453b2eb04'),
        ]
    """
    if len(uuids) == 0:
        return get_zero_uuid()
    elif len(uuids) == 1:
        return uuids[0]
    else:
        if not ordered:
            uuids = sorted(uuids)
        sep = six.b(str('-'))
        pref = six.b(str(salt) + str(sep) + str(len(uuids)))
        combined_bytes = pref + sep.join([u.bytes for u in uuids])
        combined_uuid = hashable_to_uuid(combined_bytes)
        return combined_uuid


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
        python3 -m utool.util_hash --test-hashable_to_uuid:1
        python2 -m utool.util_hash --test-hashable_to_uuid:3
        python2 -m utool.util_hash --test-hashable_to_uuid:1
        python3 -m utool.util_hash --test-hashable_to_uuid:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> hashable_ = 'foobar'
        >>> uuid_ = hashable_to_uuid(hashable_)
        >>> result = str(uuid_)
        >>> print(result)
        8843d7f9-2416-211d-e9eb-b963ff4ce281

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> hashable_ = 'foobar'.encode('utf-8')
        >>> uuid_ = hashable_to_uuid(hashable_)
        >>> result = str(uuid_)
        >>> print(result)
        8843d7f9-2416-211d-e9eb-b963ff4ce281

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> hashable_ = u'foobar'
        >>> uuid_ = hashable_to_uuid(hashable_)
        >>> result = str(uuid_)
        >>> print(result)
        8843d7f9-2416-211d-e9eb-b963ff4ce281

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from utool.util_hash import *  # NOQA
        >>> hashable_ = 10
        >>> uuid_ = hashable_to_uuid(hashable_)
        >>> result = str(uuid_)
        >>> print(result)
        e864ece8-8880-43b6-7277-c8b2cefe96ad

    """
    print('hashable_ = %r' % (hashable_,))
    # Hash the bytes
    if six.PY3:
        # If hashable_ is text (python3)
        if isinstance(hashable_, bytes):
            bytes_ = hashable_
        if isinstance(hashable_, str):
            bytes_ = hashable_.encode('utf-8')
            #print('sbytes=%r' % (bytes_,))
        elif isinstance(hashable_, int):
            bytes_ = hashable_.to_bytes(4, byteorder='big')
        else:
            #bytes_ = bytearray(hashable_)
            bytes_ = hashable_
            # bytes_ = hashable_
            #print('bytes_=%r' % (bytes_,))
    elif six.PY2:
        # If hashable_ is data (python2)
        if isinstance(hashable_, bytes):
            bytes_ = hashable_
        elif isinstance(hashable_, str):
            bytes_ = hashable_.encode('utf-8')
        elif isinstance(hashable_, int):
            import struct
            bytes_ = struct.pack('>i', hashable_)
        else:
            bytes_ = bytes(hashable_)
    print('bytes_ = %r' % (bytes_,))
    bytes_sha1 = hashlib.sha1(bytes_)
    # Digest them into a hash
    #hashstr_40 = img_bytes_sha1.hexdigest()
    #hashstr_32 = hashstr_40[0:32]
    hashbytes_20 = bytes_sha1.digest()
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


def deterministic_uuid(hashable):
    return hashable_to_uuid(hashable)


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
del ALPHABET_41
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
