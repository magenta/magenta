"""Config for Fashion-MNIST with nlatent=64.
"""

# pylint:disable=invalid-name

from functools import partial

import nn

n_latent = 64

Encoder = partial(nn.EncoderMNIST, n_latent=n_latent)
Decoder = nn.DecoderMNIST
Classifier = nn.DFull

config = {
    'Encoder': Encoder,
    'Decoder': Decoder,
    'Classifier': Classifier,
    'n_latent': n_latent,
    'dataset': 'FASHION-MNIST',
    'img_width': 28,
    'crop_width': 108,
    'batch_size': 512,
    'beta': 1.0,
    'x_sigma': 0.1,
}
