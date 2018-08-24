"""Config for MNIST classifier.
"""

# pylint:disable=invalid-name

from magenta.models.latent_transfer import nn
from magenta.models.latent_transfer.configs import mnist_0_nlatent64

config = mnist_0_nlatent64.config

Classifier = nn.DFull

config['batch_size'] = 256
config['Classifier'] = Classifier
