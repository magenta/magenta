import os
import sys
import time
import copy
import h5py
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
from IPython.display import clear_output, Image, display, HTML

#from classvis import *
from classvis import *

# download InceptionV5 model if not
if not os.path.exists("./inception5h.zip"):
    os.system("python -m wget -o ./inception5h.zip https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip")

# importing InceptionV5 model
with tf.gfile.FastGFile('tensorflow_inception_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = t_input-imagenet_mean
tf.import_graph_def(graph_def, {'input':t_preprocessed})

octaves = [{
    'scale': 1.0,
    'iter_n': 190,
    'start_sigma': .44,
    'end_sigma': 0.44,
    }, {
    'scale': 1.2,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.44,
    }, {
    'scale': 1.2,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.44,
    }, {
    'scale': 1.0,
    'iter_n': 10,
    'start_sigma': 0.44,
    'end_sigma': 0.44,
    }]

# import graph again so tf.default_graph() fetches the correct graph for visualization
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# deepdream visualization
layer = "import/softmax2_pre_activation"

background_color = np.float32([10, 90, 140])
im = np.expand_dims(imresize(imread(os.path.join("./sample_images", "images.jpg")), (224, 224)), axis = 0)

start = time.time()
# api call
is_success = class_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im},
                                     layer=layer, classes = [2],path_logdir="./Log/Inception5",
                                     path_outdir="./Output/Inception5", background_color=background_color, octaves=octaves)
start = time.time() - start
print("Total Time = %f" % (start))
