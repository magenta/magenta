"""Config for MNIST classifier.
"""

# pylint:disable=invalid-name

import nn
from configs import mnist_0_nlatent64

config = mnist_0_nlatent64.config

Classifier = nn.DFull

config['batch_size'] = 256
config['Classifier'] = Classifier
