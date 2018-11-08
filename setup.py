from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

def remove_dir(dirpath):
	if os.path.exists(dirpath) and os.path.isdir(dirpath):
		  shutil.rmtree(dirpath)
		  
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requires = [] #during runtime
tests_require=['pytest>=2.3'] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='ngene',
    version='0.1.0',
    description='Ngene is Deep Learinng provider.',
    author='Alireza',
    url='https://github.com/vafaeiar/ngene',
    packages=find_packages(PACKAGE_PATH, "ngene"),
    package_dir={'ngene': 'src'},
    include_package_data=True,
    install_requires=requires,
    license='MIT',
    zip_safe=False,
    keywords='ngene',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ]
)

remove_dir('build')
remove_dir('ngene.egg-info')
remove_dir('dist')
