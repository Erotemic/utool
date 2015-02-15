from __future__ import absolute_import, division, print_function
import six


def ensure_iterable(obj):
    if isiterable(obj):
        return obj
    else:
        return [obj]


def isiterable(obj):
    """
    Returns if the object can be iterated over and is NOT a string
    # TODO: implement isscalar similar to numpy
    """
    try:
        iter(obj)
        return not isinstance(obj, six.string_types)
    except:
        return False
    #return np.iterable(obj) and not isinstance(obj, six.string_types)


#def isscalar(obj):
