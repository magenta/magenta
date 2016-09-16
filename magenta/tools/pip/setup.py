# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages


_VERSION = '0.1.0'

REQUIRED_PACKAGES = [
    'wheel',
    'pretty_midi >= 0.2.5',
    'tensorflow >= 0.10.0',
]

setup(
    name='magenta',
    version=_VERSION,
    description='Use machine learning to create art and music',
    long_description='',
    url='https://magenta.tensorflow.org/',
    author='Google Inc.',
    author_email='opensource@google.com',
    license='Apache 2',
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
    keywords='tensorflow machine learning magenta music art',

    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)

