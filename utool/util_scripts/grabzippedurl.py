#!/usr/bin/env python2.7
"""
downloads an archive file and then unzips it into a directory with the same name
as the archive (sans the .zip or .tar.gz)
"""
from __future__ import absolute_import, division, print_function
import sys
import utool

if __name__ == '__main__':
    url = sys.argv[1]
    download_dir = '.'
    utool.grab_zipped_url(url, download_dir=download_dir, cleanup=False)
