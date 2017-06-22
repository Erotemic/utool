# -*- coding: utf-8 -*-
"""
TODO: in the future maybe this is a contribution to pip
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import utool as ut


# def get_site_package_directories():
#     import site
#     import sys
#     import six
#     sitepackages = site.getsitepackages()
#     if sys.platform.startswith('darwin'):
#         if six.PY2:
#             macports_site = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages'
#         else:
#             macports_site = '/opt/local/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages'
#             assert six.PY2, 'fix this for python 3'
#         sitepackages = [macports_site] + sitepackages
#     return sitepackages


def module_stdinfo_dict(module, versionattr='__version__', version=None,
                        libdep=None, name=None, **kwargs):
    infodict = ut.odict()
    if module is None:
        infodict['__version__'] = version
        infodict['__name__'] = name
        infodict['__file__'] = 'None'
    else:
        if version is not None:
            infodict['__version__'] = version
        else:
            infodict['__version__'] = getattr(module, versionattr, None)
        infodict['__name__'] = name
        infodict['__name__'] = getattr(module, '__name__', None)
        infodict['__file__'] = getattr(module, '__file__', None)

    if libdep is not None:
        infodict['libdep'] = libdep
    infodict.update(kwargs)
    return infodict


def print_module_info(modname):
    print('Checking modname = %r' % (modname,))
    # Handles special cases for certain modules
    if modname.lower() == 'pillow':
        from PIL import Image
        import PIL
        pil_path = PIL.__path__
        infodict = module_stdinfo_dict(Image, versionattr='PILLOW_VERSION',
                                       image_version=Image.VERSION,
                                       pil_path=pil_path)
    elif modname.lower() == 'pyqt4':
        from PyQt4 import QtCore
        infodict = module_stdinfo_dict(QtCore, 'PYQT_VERSION_STR')
    elif modname.lower() == 'pyqt5':
        from PyQt5 import QtCore
        infodict = module_stdinfo_dict(QtCore, 'PYQT_VERSION_STR')
    else:
        # Handle normal modules
        module = ut.import_modname(modname)
        infodict = module_stdinfo_dict(module)
    if any([infodict['__file__'].endswith(ext) for ext in ut.LIB_EXT_LIST]):
        infodict['libdep'] = ut.get_dynlib_dependencies(infodict['__file__'])
    return print(ut.repr4(infodict, strvals=True))


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/utool/utool/util_scripts
        python ~/code/utool/utool/util_scripts/pipinfo.py
        python ~/code/utool/utool/util_scripts/pipinfo.py --allexamples

        pipinfo.py networkx
        pipinfo.py sklearn
        pipinfo.py cv2
        pipinfo.py PyQt4
    """
    modname = ut.get_argval('--modname')
    if not modname:
        args = ut.get_cmdline_varargs()
        if len(args) > 0:
            modname = args[0]
    assert modname
    print_module_info(modname)
