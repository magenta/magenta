# Copyright 2020 The Magenta Authors.
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

"""Config for MNIST with nlatent=64.
"""

# pylint:disable=invalid-name

import functools

from magenta.models.latent_transfer import nn

n_latent = 100

Encoder = functools.partial(nn.EncoderMNIST, n_latent=n_latent)
Decoder = nn.DecoderMNIST
Classifier = nn.DFull

config = {
    'Encoder': Encoder,
    'Decoder': Decoder,
    'Classifier': Classifier,
    'n_latent': n_latent,
    'dataset': 'MNIST',
    'img_width': 28,
    'crop_width': 108,
    'batch_size': 512,
    'beta': 1.0,
    'x_sigma': 0.1,
}
