import subprocess
from distutils.core import setup, Extension

VERSION = open('version').read().strip()
GIT_VERSION = subprocess.check_output("git describe --always", shell=True)

pykgraph = Extension('pykgraph',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++11', '-msse2', '-fopenmp', '-DKGRAPH_VERSION=%s' % GIT_VERSION],
        include_dirs = ['/usr/local/include', '.'],
        libraries = ['boost_python', 'boost_timer', 'boost_chrono', 'gomp'],
        sources = ['kgraph.cpp', 'metric.cpp', 'python/pykgraph.cpp'],
        depends = ['kgraph.h', 'kgraph-data.h'])

setup (name = 'pykgraph',
       version = '2.0',
       url = 'https://github.com/aaalgo/kgraph',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'Approximate K-NN search',
       ext_modules = [pykgraph]
       )
