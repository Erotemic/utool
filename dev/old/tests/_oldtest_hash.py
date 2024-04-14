#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REMEMBER In Python 3 think: text or data.
str.encode:   text -> data
bytes.decode: data -> text
"""
import utool
import six
from os.path import join
#import uuid


lorium_text = '''
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. '''


def test_file_hash():
    resdir = utool.get_app_resource_dir('utool')
    test_fpath = join(resdir, 'lorium_ipsum.txt')
    if not utool.checkpath(test_fpath, verbose=True, n=100):
        utool.write_to(test_fpath, lorium_text)
    hash_ = utool.get_file_hash(test_fpath)
    target_hash_ = b'\xd1Y\xe5\xa2\xc1\xd8\xb8\nS\xb1?\x16\xfe\xc5\x88\xbd\x9e\xb4\xe3\xda'
    print(repr(hash_))
    print(repr(target_hash_))
    assert hash_ == target_hash_


def test_hashstr():
    hashstr_ = utool.hashstr(lorium_text)
    print(repr(hashstr_))
    target_hashstr = 'yy7@rnyuhe&zhj0k'
    print(repr(target_hashstr))
    assert hashstr_ == target_hashstr


def test_hashstr_components():

    import hashlib
    print('lorium_text = %r' % (lorium_text,))
    data = lorium_text.encode('utf-8')
    print('data = %r' % (data,))
    hashobj = hashlib.sha512(data)
    print('hashobj = %r' % (hashobj,))
    hashstr = hashobj.hexdigest()
    print('hashstr = %r' % (hashstr,))
    assert hashstr == ('c48e65cb600a078819bbfa227b8c02ee5e198cfe9ebc4eb6791f'
                       '69103bb11bc4b2a685c67f9f09ca3a1f364732cef0b2f36a745b'
                       'ae2b02bd1972592eeb402bd2')
    hashstr2 = utool.convert_hexstr_to_bigbase(hashstr)
    print('hashstr2 = %r' % (hashstr2,))
    assert hashstr2 == 'yy7@rnyuhe&zhj0kd+g&22imak39piwuq47t06dinfer0j7ye&l4mr!gl89!uj8v0idfjqy0pry972pu@ety8f8n7b8%0ob6'


def test_augment_uuid():
    # TODO: This is still divergent between python2 and python3
    uuid_ = utool.get_zero_uuid()
    #uuid_ = uuid.uuid1()

    #uuidhex_data = uuid_.get_bytes()
    uuidhex_data = uuid_.bytes
    print('uuidhex_data = %r' % uuidhex_data)

    hashable_text1 = '[0, 0, 100, 100]'
    hashable_text2 = ''
    if six.PY3:
        hashable_data1 = hashable_text1.encode('utf-8')
        hashable_data2 = hashable_text2.encode('utf-8')
    else:
        hashable_data1 = hashable_text1
        hashable_data2 = hashable_text2
    augmented_data1 = uuidhex_data + hashable_data1
    augmented_data2 = uuidhex_data + hashable_data2

    augmented_uuid1_ = utool.hashable_to_uuid(augmented_data1)
    augmented_uuid2_ = utool.hashable_to_uuid(augmented_data2)

    print('augmented_data1 =%r' % augmented_data1)
    print('augmented_data2 =%r' % augmented_data2)

    struuid_ = utool.hashable_to_uuid(str(uuid_))

    print('           uuid_=%r' % (uuid_,))
    print('augmented_uuid1_=%r' % (augmented_uuid1_,))
    print('augmented_uuid2_=%r' % (augmented_uuid2_,))
    print('hash2uuid(uuid_)=%r' % (struuid_,))

    # Test what is capable of being augmented
    import numpy as np
    augmented_uuid_T1 = utool.augment_uuid(uuid_, hashable_data1)
    augmented_uuid_T2 = utool.augment_uuid(uuid_, hashable_data2)
    augmented_uuid_T3 = utool.augment_uuid(uuid_, hashable_data2, hashable_data1)
    augmented_uuid_T4 = utool.augment_uuid(uuid_, hashable_data1, hashable_data2)
    augmented_uuid_T5 = utool.augment_uuid(uuid_, [1, 2, 3])
    augmented_uuid_T5 = utool.augment_uuid(uuid_, (1, 2, 3))
    augmented_uuid_T6 = utool.augment_uuid(uuid_, np.array((1, 2, 3)))

    print('augmented_uuid_T1=%r' % (augmented_uuid_T1,))
    print('augmented_uuid_T2=%r' % (augmented_uuid_T2,))
    print('augmented_uuid_T3=%r' % (augmented_uuid_T3,))
    print('augmented_uuid_T4=%r' % (augmented_uuid_T4,))
    print('augmented_uuid_T5=%r' % (augmented_uuid_T5,))
    print('augmented_uuid_T6=%r' % (augmented_uuid_T6,))


def test_byteslike():
    text = lorium_text[0:11]
    print('type(text) = %r' % type(text))
    data = text.encode('utf-8')
    print('type(data) = %r' % type(data))
    memview = memoryview(data)
    print('type(memview) = %r, memview=%r' % (type(memview), memview))


if __name__ == '__main__':
    test_byteslike()
    utool.run_test(test_augment_uuid)
    import sys
    print(sys.executable)
