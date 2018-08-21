"""Config for Fasion-MNIST classifier.
"""

# pylint:disable=invalid-name

import nn
from configs import fashion_mnist_0_nlatent64

config = fashion_mnist_0_nlatent64.config

Classifier = nn.DFull

config['batch_size'] = 256
config['Classifier'] = Classifier
