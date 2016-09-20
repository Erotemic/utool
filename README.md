|  **`Travis CI Status`**   |
|-------------------|
|[![Travis](https://img.shields.io/travis/Erotemic/utool.svg)](https://travis-ci.org/Erotemic/utool)|

.. image:: https://img.shields.io/pypi/v/utool.svg
        :target: https://pypi.python.org/pypi/utool


utool
=====

Useful Utility Tools For You!

Documenation: http://erotemic.github.io/utool

The most useful functions in this library are:

* ut.flatten
* ut.take
* ut.take_column
* ut.compress
* ut.ichunks
* ut.itertwo
* ut.isiterable
* ut.group_items
* ut.dict_subset
* ut.dict_hist
* ut.map_dict_vals
* ut.map_dict_keys
* ut.memoize
* ut.get_argflag
* ut.get_argval
* ut.ProgIter
* ut.MemoryTracker
* ut.InteractiveIter
* ut.color_print
* ut.ensuredir
* ut.glob
* ut.grep
* ut.sed
* ut.ls
* ut.repr2

Utool is a collection of tools that I've found useful. I've written most of them
from scratch, but there are a few I've taken or partially taken from
stackoverflow. References are given in most locations. Some of the older parts
of the module may not have it, but I fix a missed reference whenever I see it.

The module contains mostly standalone functions that could easily be
copy/pasted into your project as you see fit. I try not to introduce too many
dependencies (or at least keep them optional). But there are also somethings
that help glue together my projects (e.g. util_inject and util_logging which I
use to change my print functions into logging statements, automatically hook
into a line_profile decorator, and util_dbg which has a lot of useful
debugging commands)

Some functions may be broken, deprecated, unfinished, or undocumented. This is
because I add to this repo as I need to. That being said, for the time being,
everything in the package is subject to change, but I probably wont change
something unless it's bad (there are plenty of functions in here that are bad,
inefficient, or poorly named, but there are some pretty cool ones too).

There are plenty of things which are perfectly universal (util_str, util_list,
util_iter, util_time, util_arg, ...), but there are things which are custom to my build
environment (lots of things in util_git, util_setup, and some in util_scripts). 

My ultimate goal for this would be to added to pypi. Some steps that would need
to be taken before that would be: eliminate me-sepcific code and introduce some
configuration file, clean up deprecated or useless functions, reorganize
functions, introduce more documentation and doctests, and clean up all the (or
most of the) TODOs and FIXMEs.

Utool is released under the Apache License Version 2.0

If the pypi release is currently broken. Installing utool can be done via pip
and github. Generally the next branch has the latest and greatest.

pip install git+https://github.com/Erotemic/utool.git@next
