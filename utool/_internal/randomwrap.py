"""This is a minimal Python client for Mads Haahr's random number generator at www.random.org

# This tiny set of functions only implements a subset of the HTTP interface available. In particular it only uses the 'live'
# random number generator, and doesn't offer the option of using the alternative 'stored' random
# number sets. However, it should be obvious how to extend it by sending requests with different parameters.
# The web service code is modelled on Mark Pilgrim's "Dive into Python" tutorial at http://www.diveintopython.org/http_web_services
# This client by George Dunbar, University of Warwick (Copyright George Dunbar, 2008)
# It is distributed under the Gnu General Public License.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See <http://www.gnu.org/licenses/> for a copy of the GNU General Public License.
    For use that falls outside this license, please contact me.


To use in a python script or at the interactive prompt
(randomwrapy.py has to be in the Python search path, of course):

from randomwrapy import *

rnumlistwithoutreplacement(0, 12)
   # returns a list of the integers 0 - 12 inclusive, in a random order

rnumlistwithreplacement(12, 5)
   # returns 12 integers from the range [0, 5]

rnumlistwithreplacement(12, 5, 2)
   # returns 12 integers from the range [2, 5]

rrandom()
   # returns a random float in the range [0, 1]

reportquota()
   # tells you how many bits you have available; visit www.random.org/quota for more information

Arguments where given are (must be) numbers, of course.
There is almost no error checking in these scripts! For example, if
the web site is down, Python will simply raise an exception and report the
http error code. See worldrandom.py for an alternative implementation
that goes a little further with error checking.

"""

from six.moves import urllib


def rnumlistwithoutreplacement(min, max):
    """Returns a randomly ordered list of the integers between min and max"""
    if checkquota() < 1:
        raise Exception("Your www.random.org quota has already run out.")
    requestparam = build_request_parameterNR(min, max)
    request = urllib.request.Request(requestparam)
    request.add_header('User-Agent', 'randomwrapy/0.1 very alpha')
    opener = urllib.request.build_opener()
    numlist = opener.open(request).read()
    return numlist.split()

# helper


def build_request_parameterNR(min, max):
    randomorg = 'http://www.random.org/sequences/?min='
    vanilla = '&format=plain&rnd=new'
    params = str(min) + '&max=' + str(max)
    return randomorg + params + vanilla


def rnumlistwithreplacement(howmany, max, min=0):
    """Returns a list of howmany integers with a maximum value = max.
    The minimum value defaults to zero."""
    if checkquota() < 1:
        raise Exception("Your www.random.org quota has already run out.")
    requestparam = build_request_parameterWR(howmany, min, max)
    request = urllib.request.Request(requestparam)
    request.add_header('User-Agent', 'randomwrapy/0.1 very alpha')
    opener = urllib.request.build_opener()
    numlist = opener.open(request).read()
    return numlist.split()

"""
Example usage:

Roll a dice 12 times (returning integers in the range [0,5]):
  rnumlistwithreplacement(12, 5)

Roll a dice 12 times (returning integers in the more familiar range [1,6]):
  rnumlistwithreplacement(12, 6, 1)
"""

# helper


def build_request_parameterWR(howmany, min, max):
    randomorg = 'http://www.random.org/integers/?num='
    vanilla = '&col=1&base=10&format=plain&rnd=new'
    params = str(howmany) + '&min=' + str(min) + '&max=' + str(max)
    return randomorg + params + vanilla

# next function is prototype for integration with random module of python
# see worldrandom module for a more developed implementation


def rrandom():
    """Get the next random number in the range [0.0, 1.0].
    Returns a float."""
    import urllib.request
    import urllib.error
    import urllib.parse
    if checkquota() < 1:
        raise Exception("Your www.random.org quota has already run out.")
    request = urllib.request.Request(
        'http://www.random.org/integers/?num=1&min=0&max=1000000000&col=1&base=10&format=plain&rnd=new')
    request.add_header('User-Agent', 'randomwrapy/0.1 very alpha')
    opener = urllib.request.build_opener()
    numlist = opener.open(request).read()
    num = numlist.split()[0]
    return float(num) / 1000000000


def checkquota():
    request = urllib.request.Request(
        "http://www.random.org/quota/?format=plain")
    request.add_header('User-Agent', 'randomwrapy/0.1 very alpha')
    opener = urllib.request.build_opener()
    quota = opener.open(request).read()
    return int(quota)


def reportquota():
    request = urllib.request.Request(
        "http://www.random.org/quota/?format=plain")
    request.add_header('User-Agent', 'randomwrapy/0.1 very alpha')
    opener = urllib.request.build_opener()
    quota = opener.open(request).read()
    print("This IP address has", quota,
          "bits left. Visit http://www.random.org/quota for more information.")
