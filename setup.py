#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from os.path import join
from sys import exit, prefix, version_info, argv

#if 'install' in argv: #-----------------------------------
#    print("Running source install...")
#    import install
#    install.main()
#    print('Done!')

# pip/setuptools install ------------------------------

classifiers = """\
Development Status :: 3 - Whatever
Intended Audience :: Scientists
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3
Topic :: Utilities
"""


# check python version
ver = (version_info.major, version_info.minor)
assert ver >= (3, 6), 'fitlib uses f-strings, which are available starting from Python 3.6'



setup(name='fitlib',
      version='0.1',
      description="A library to fit function with arbitrary parameter types",
      long_description="A library that allows fitting not only function with a fixed number of scalar parameters, but essentially arbitrary functions.",
      author='Timofey',
      author_email='Timofey.Balashov@kit.edu',
      platforms='POSIX',
      keywords=['Curve fitting'],
      classifiers=[clsf for clsf in classifiers.split('\n') if clsf],
      license='BSD',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
      packages=find_packages(),
)

