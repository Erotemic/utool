[![Travis](https://img.shields.io/travis/Erotemic/utool.svg)](https://travis-ci.org/Erotemic/utool)
[![Pypi](https://img.shields.io/pypi/v/utool.svg)](https://pypi.python.org/pypi/utool)


# utool

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

# Documenation
http://erotemic.github.io/utool


# Installation
Installation can now be done via pypi

    pip install utool

If the pypi release is currently broken. Installing utool can be done via pip
and github. Generally the next branch has the latest and greatest.

    pip install git+https://github.com/Erotemic/utool.git@next

Utool is released under the Apache License Version 2.0
