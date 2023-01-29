|Pypi| |Downloads| |Codecov| |GithubActions|

utool
-----

Notice: This is a "kitchen sink" library. While it is still somewhat maintained, I am trying to "end-of-life" it as soon as possible. I have refactored it into several other projects. Please see https://github.com/Erotemic/ubelt for a well-maintained curated collection of utilities. Also see https://github.com/Erotemic/xdev for ports of the more developer-y functions in this libray.

This project is only maintained for the benefit of
https://github.com/Erotemic/ibeis and is being phased whenever possible.

----

Useful Utility Tools For You!

The `utool` library is a collection of tools that I've found useful. I've
written most of them from scratch, but there are a few I've taken or partially
taken from StackOverflow. References are given in most locations.  

In my experience the most useful functions in this library are:

* `ut.flatten`
* `ut.take`
* `ut.take_column`
* `ut.compress`
* `ut.ichunks`
* `ut.itertwo`
* `ut.isiterable`
* `ut.group_items`
* `ut.dict_subset`
* `ut.dict_hist`
* `ut.map_dict_vals`
* `ut.map_dict_keys`
* `ut.memoize`
* `ut.get_argflag`
* `ut.get_argval`
* `ut.ProgIter`
* `ut.Timer`
* `ut.Timerit`
* `ut.MemoryTracker`
* `ut.InteractiveIter`
* `ut.color_print`
* `ut.ensuredir`
* `ut.glob`
* `ut.grep`
* `ut.sed`
* `ut.ls`
* `ut.repr2`

Warning: This module does contain personalized functions that help glue
together my projects.  Over time these will be removed.  While many of the
functions in this repo are mature, the repo itself is a work in progress.  Some
functions may be broken, deprecated, unfinished, or undocumented. Thus, some
APIs may refactored or removed in the future.


Documenation
------------
http://erotemic.github.io/utool


Installation
--------------
Installation can now be done via pypi

.. code:: bash

    pip install utool



.. |CircleCI| image:: https://circleci.com/gh/Erotemic/utool.svg?style=svg
    :target: https://circleci.com/gh/Erotemic/utool
.. |Travis| image:: https://img.shields.io/travis/Erotemic/utool/main.svg?label=Travis%20CI
   :target: https://travis-ci.org/Erotemic/utool?branch=main
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/github/Erotemic/utool?branch=main&svg=True
   :target: https://ci.appveyor.com/project/Erotemic/utool/branch/main
.. |Codecov| image:: https://codecov.io/github/Erotemic/utool/badge.svg?branch=main&service=github
   :target: https://codecov.io/github/Erotemic/utool?branch=main
.. |Pypi| image:: https://img.shields.io/pypi/v/utool.svg
   :target: https://pypi.python.org/pypi/utool
.. |Downloads| image:: https://img.shields.io/pypi/dm/utool.svg
   :target: https://pypistats.org/packages/utool
.. |ReadTheDocs| image:: https://readthedocs.org/projects/utool/badge/?version=latest
    :target: http://utool.readthedocs.io/en/latest/
.. |GithubActions| image:: https://github.com/Erotemic/utool/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/Erotemic/utool/actions?query=branch%3Amain
