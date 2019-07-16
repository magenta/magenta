# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A setuptools based setup module for magenta.music."""

import sys

from setuptools import find_packages
from setuptools import setup

# Bit of a hack to parse the version string stored in version.py without
# executing __init__.py, which will end up requiring a bunch of dependencies to
# execute (e.g., tensorflow, pretty_midi, etc.).
# Makes the __version__ variable available.
with open('magenta/music/version.py') as in_file:
  exec(in_file.read())  # pylint: disable=exec-used

if '--gpu' in sys.argv:
  gpu_mode = True
  sys.argv.remove('--gpu')
else:
  gpu_mode = False

REQUIRED_PACKAGES = [
    'IPython',
    'absl-py',
    'backports.tempfile',
    'bokeh >= 0.12.0',
    'intervaltree >= 2.1.0',
    'librosa >= 0.6.2',
    'mido == 1.2.6',
    'numpy >= 1.14.6',  # 1.14.6 is required for colab compatibility.
    'pandas >= 0.18.1',
    'pretty_midi >= 0.2.6',
    'protobuf >= 3.6.1',
    'scipy >= 0.18.1, <= 1.2.0',  # 1.2.1 causes segfaults in pytest.
    'six >= 1.12.0',
    'wheel',
    'futures;python_version=="2.7"',
    'apache-beam[gcp] >= 2.8.0;python_version=="2.7"',
]

if gpu_mode:
  REQUIRED_PACKAGES.append('tensorflow-gpu >= 1.0.0')
else:
  REQUIRED_PACKAGES.append('tensorflow >= 1.0.0')

CONSOLE_SCRIPTS = []

setup(
    name='magenta.music-gpu' if gpu_mode else 'magenta.music',
    version=__version__,  # pylint: disable=undefined-variable
    description='Music processing library from Magenta',
    long_description='',
    url='https://magenta.tensorflow.org/',
    author='Google Inc.',
    author_email='magenta-discuss@gmail.com',
    license='Apache 2',
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'Topic :: Multimedia :: Sound/Audio :: Analysis'
    ],
    keywords='magenta music mir midi audio musicxml',

    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': ['%s = %s:console_entry_point' % (n, p) for n, p in
                            ((s.split('.')[-1], s) for s in CONSOLE_SCRIPTS)],
    },

    setup_requires=['pytest-runner', 'pytest-pylint'],
    tests_require=[
        'pytest',
        'pylint < 2.0.0;python_version<"3"',
        # pylint 2.3.0 and astroid 2.2.0 caused spurious errors,
        # so lock them down to known good versions.
        'pylint == 2.2.2;python_version>="3"',
        'astroid == 2.0.4;python_version>="3"',
    ],
)
