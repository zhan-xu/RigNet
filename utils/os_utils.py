#-------------------------------------------------------------------------------
# Name:        mst_utils.py
# Purpose:     utilize functions for system commands
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

from __future__ import absolute_import

import os
import errno

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def isfile(fname):
    return os.path.isfile(fname)

def isdir(dirname):
    return os.path.isdir(dirname)

def join(path, *paths):
    return os.path.join(path, *paths)