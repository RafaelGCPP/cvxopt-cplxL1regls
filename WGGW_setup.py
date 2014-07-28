# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:53:53 2014

@author: Rafael
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'WGGW_C',
  ext_modules = cythonize("WGGW_C.pyx"),
  include_dirs=[numpy.get_include()]
)