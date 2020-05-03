|Pypi| |Downloads| |Codecov| |Travis| |Appveyor| 

utool
-----

Notice: This is a "kitchen sink" library. While it is still somewhat maintained, I am trying to "end-of-life" it as soon as possible. I have refactored it into several other projects. Please see https://github.com/Erotemic/ubelt for a well-maintained curated collection of utilities. Also see https://github.com/Erotemic/xdev for ports of the more developer-y functions in this libray.

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

If the pypi release is currently broken. Installing utool can be done via pip
and github. 

.. code:: bash

    pip install git+https://github.com/Erotemic/utool.git@master

Utool is released under the Apache License Version 2.0


.. |CircleCI| image:: https://circleci.com/gh/Erotemic/utool.svg?style=svg
    :target: https://circleci.com/gh/Erotemic/utool
.. |Travis| image:: https://img.shields.io/travis/Erotemic/utool/master.svg?label=Travis%20CI
   :target: https://travis-ci.org/Erotemic/utool?branch=master
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/github/Erotemic/utool?branch=master&svg=True
   :target: https://ci.appveyor.com/project/Erotemic/utool/branch/master
.. |Codecov| image:: https://codecov.io/github/Erotemic/utool/badge.svg?branch=master&service=github
   :target: https://codecov.io/github/Erotemic/utool?branch=master
.. |Pypi| image:: https://img.shields.io/pypi/v/utool.svg
   :target: https://pypi.python.org/pypi/utool
.. |Downloads| image:: https://img.shields.io/pypi/dm/utool.svg
   :target: https://pypistats.org/packages/utool
.. |ReadTheDocs| image:: https://readthedocs.org/projects/utool/badge/?version=latest
    :target: http://utool.readthedocs.io/en/latest/
