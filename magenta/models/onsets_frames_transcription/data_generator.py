import sys

import numpy as np

import threading

import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

if FLAGS.using_plaidml:
    import keras
else:
    import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, batch_size, steps_per_epoch, shuffle=False, use_numpy=True):
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.shuffle = shuffle
        self.use_numpy = use_numpy
        self.iterator = iter(self.dataset)
        self.lock = threading.Lock()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 0

    def __getitem__(self, index):
        'Generate one batch of data'
        with self.lock:
            if self.use_numpy:
                x, y = ([t.numpy() for t in tensors] for tensors in next(self.iterator))
            else:
                x, y = ([t for t in tensors] for tensors in next(self.iterator))
            return x, y

    def get(self):
        return self.__getitem__(0)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(np.arange(self.steps_per_epoch * self.batch_size))