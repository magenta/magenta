import os
import sys
import time
import copy
import h5py
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf

from visualize_utils import *

flags = tf.flags
flags.DEFINE_string("model", "tensorflow_inception_graph.pb", "Model")
flags.DEFINE_string("layer", "import/softmax2_pre_activation", "Layer name")
flags.DEFINE_string("params_path", ".", "Path to folder containing python file 'params.py'")
flags.DEFINE_string("logdir_path", "./Log/Inception5", "Path to folder where logs will be stored")
flags.DEFINE_string("outdir_path", "./Output/Inception5", "Path to folder where generated images will be stored")
flags.DEFINE_boolean("use_bilateral", True, "Flag for whether or not to use bilateral filtering. Params for bilateral filtering should be set for each octave defined in 'params.py'")
flags.DEFINE_boolean("use_tv_bregman", False, "Flag for whether or not to perform total-variation denoising")
FLAGS = flags.FLAGS

sys.path.insert(0, FLAGS.params_path)
from params import *

# download InceptionV5 model if not
if not os.path.exists("./inception5h.zip"):
    print("Downloading InceptionV5")
    os.system("python -m wget -o ./inception5h.zip https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip")

# importing InceptionV5 model
with tf.gfile.FastGFile('tensorflow_inception_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = t_input-imagenet_mean
tf.import_graph_def(graph_def, {'input':t_preprocessed})



# import graph again so tf.default_graph() fetches the correct graph for visualization
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# deepdream visualization
layer = FLAGS.layer

# Set initial image of noise to visualize on
im = np.random.normal(background_color, 8, (224, 224, 3))
im = np.expand_dims(im, axis = 0)

start = time.time()

is_success = class_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im},
                                     layer=layer, classes=classes, path_logdir=FLAGS.logdir_path,
                                     path_outdir=FLAGS.outdir_path, background_color=background_color, octaves=octaves, use_bilateral=FLAGS.use_bilateral, use_tv_bregman=FLAGS.use_tv_bregman)
start = time.time() - start
print("Total Time = %f" % (start))
