#!/usr/bin/env python3
import os, sys
import numpy as np
from glob import glob
import subprocess as sp
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

include_dirs = [np.get_include(),
        '/usr/local/include',
        '3rd/pybind11/include',
        '3rd/xtl/include',
        '3rd/xtensor/include',
        '3rd/xtensor-python/include']
library_dirs = []
libraries = [
        'boost_timer'
        ]

ext = Extension('pykgraph',
        language = 'c++',
        extra_compile_args = ['-std=c++17', '-O3', '-g', '-Wno-sign-compare', '-Wno-parentheses', '-DDEBUG', '-Wno-narrowing', '-Wno-attributes', '-Wno-unknown-pragmas'],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        libraries = libraries,
        sources = ['python-api.cpp', 'kgraph.cpp', 'metric.cpp']
        )

setup (name = 'pykgraph',
       version = '0.0.1',
       author = 'Wei Dong',
       author_email = 'wdong@aaalgo.com',
       license = 'BSD',
       description = '',
       ext_modules = [ext],
       )

