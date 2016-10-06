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
"""A setuptools based setup module for magenta."""

from setuptools import find_packages
from setuptools import setup

import magenta


REQUIRED_PACKAGES = [
    'mido >= 1.1.17',
    'pretty_midi >= 0.2.6',
    'tensorflow >= 0.10.0',
    'wheel',
]

CONSOLE_SCRIPTS = [
    'magenta.interfaces.midi.magenta_midi',
    'magenta.models.attention_rnn.attention_rnn_create_dataset',
    'magenta.models.attention_rnn.attention_rnn_generate',
    'magenta.models.attention_rnn.attention_rnn_train',
    'magenta.models.basic_rnn.basic_rnn_create_dataset',
    'magenta.models.basic_rnn.basic_rnn_generate',
    'magenta.models.basic_rnn.basic_rnn_train',
    'magenta.models.lookback_rnn.lookback_rnn_create_dataset',
    'magenta.models.lookback_rnn.lookback_rnn_generate',
    'magenta.models.lookback_rnn.lookback_rnn_train',
    'magenta.scripts.convert_midi_dir_to_note_sequences',
]

setup(
    name='magenta',
    version=magenta.__version__,
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
    entry_points={
        'console_scripts': ['%s = %s:console_entry_point' % (n, p) for n, p in
                            ((s.split('.')[-1], s) for s in CONSOLE_SCRIPTS)],
    },
)

