#!/usr/bin/python
# -*- coding: utf-8 -*-
################################################
# Imports                                      #
################################################

import os
import time
from six import iteritems
import numpy as np
import tensorflow as tf
from skimage.restoration import denoise_tv_bregman, denoise_bilateral
from scipy.misc import imsave

################################################
# Global Variables                             #
################################################

# optional hyperparameter settings

config = {
    'N': 8,
    'EPS': 1e-7,
    'MAX_IMAGES': 1,
    'MAX_FEATUREMAP': 1024,
    'FORCE_COMPUTE': False,
    'TV_DENOISE_WEIGHT': 2.0,
    'NUM_LAPLACIAN_LEVEL': 4,
    }

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
K5X5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
default_octaves = [{
    'scale': 1.0,
    'iter_n': 190,
    'start_sigma': .44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.2,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.2,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.0,
    'iter_n': 10,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }]

dict_layer = {'r': 'relu', 'p': 'maxpool', 'c': 'conv2d'}

################################################
# Helper Functions for Visualizing and Writing #
################################################

# save given graph object as meta file

def _save_model(graph):
    """
....Save the given TF graph at PATH = "./model/tmp-model"

....:param graph:
........TF graph
....:type graph:  tf.Graph object

....:return:
........Path to saved graph
....:rtype: String
...."""

    PATH = os.path.join('model', 'tmp-model')
    make_dir(path=os.path.dirname(PATH))

    with graph.as_default():
        with tf.Session() as sess:
            fake_var = tf.Variable([0.0], name='fake_var')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, PATH)

    return PATH + '.meta'


# if dir not exits make one

def _is_dir_exist(path):
    return os.path.exists(path)


def make_dir(path):
    is_success = True

    # if dir is not exist make one

    if not _is_dir_exist(path):
        try:
            os.makedirs(path)
        except OSError, exc:
            is_success = False
    return is_success


# get operation and tensor by name

def get_operation(graph, name):
    return graph.get_operation_by_name(name=name)


def get_tensor(graph, name):
    return graph.get_tensor_by_name(name=name)


def parse_tensors_dict(graph, layer_name, value_feed_dict):
    x = []
    feed_dict = {}
    with graph.as_default() as g:

        # get op of name given in method argument layer_name

        op = get_operation(graph=g, name=layer_name)
        op_tensor = op.outputs[0]  # output tensor of the operation
        tensor_shape = op_tensor.get_shape().as_list()  # get shape of tensor

        # check for limit on number of feature maps

        if not config['FORCE_COMPUTE'] and tensor_shape[-1] \
            > config['MAX_FEATUREMAP']:
            print 'Skipping. Too many featuremaps. May cause memory errors.'
            return None

        # creating feed_dict and find input tensors

        X_in = None

        # find tensors of value_feed_dict
        # in current graph by name

        for (key_op, value) in iteritems(value_feed_dict):
            tmp = get_tensor(graph=g, name=key_op.name)
            feed_dict[tmp] = value
            x.append(tmp)

        X_in = x[0]
        feed_dict[X_in] = (feed_dict[X_in])[:config['MAX_IMAGES']]  # only taking first MAX_IMAGES from given images array
    return (op_tensor, x, X_in, feed_dict)


def image_normalization(image, s=0.1, ubound=255.0):
    """
....Min-Max image normalization. Convert pixle values in range [0, ubound]

....:param image:
........A numpy array to normalize
....:type image: 3-D numpy array

....:param ubound:
........upperbound for a image pixel value
....:type ubound: float (Default = 255.0)

....:return:
........A normalized image
....:rtype: 3-D numpy array
...."""

    img_min = np.min(image)
    img_max = np.max(image)
    return ((image - img_min) * ubound / (img_max - img_min
            + config['EPS'])).astype('uint8')


def _im_normlize(images, ubound=255.0):
    N = len(images)
    (H, W, C) = images[0][0].shape

    for i in range(N):
        for j in range(images[i].shape[0]):
            images[i][j] = image_normalization(images[i][j],
                    ubound=ubound)
    return images


def _write_deepdream(
    images,
    layer,
    path_outdir,
    path_logdir,
    ):

    is_success = True

    images = _im_normlize([images])
    (layer, units, k) = layer

    # write into disk

    path_out = os.path.join(path_outdir, layer.lower().replace('/', '_'
                            ))
    is_success = make_dir(path_out)

    for i in range(len(images)):
        for j in range(images[i].shape[0]):
            imsave(os.path.join(path_out, 'image_%d.png' % units[i
                   * images[i].shape[0] + j + k]), images[i][j],
                   format='png')

    # write into logfile

    path_log = os.path.join(path_logdir, layer.lower().replace('/', '_'
                            ))
    is_success = make_dir(path_log)

    with tf.Graph().as_default() as g:
        image = tf.placeholder(tf.float32, shape=[None, None, None,
                               None])

        image_summary_t = tf.summary.image(name='One_By_One_DeepDream',
                tensor=image, max_outputs=config['MAX_FEATUREMAP'])

        with tf.Session() as sess:
            summary = sess.run(image_summary_t,
                               feed_dict={image: np.concatenate(images,
                               axis=0)})
        try:
            file_writer = tf.summary.FileWriter(path_log, g)  # create file writer

            # compute and write the summary

            file_writer.add_summary(summary)
        except:
            is_success = False
            print 'Error occured in writting results into log file.'
        finally:
            file_writer.close()  # close file writer
    return is_success


def _graph_import_function(PATH):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(PATH)  # Import graph
        new_saver.restore(sess,
                          tf.train.latest_checkpoint(os.path.dirname(PATH)))
        return sess


# laplacian pyramid gradient normalization

def _lap_split(img):
    '''Split the image into lo and hi frequency components'''

    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, K5X5, [1, 2, 2, 1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, K5X5 * 4, tf.shape(img), [1,
                2, 2, 1])
        hi = img - lo2
    return (lo, hi)


def _lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''

    levels = []
    for i in range(n):
        (img, hi) = _lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]


def _lap_merge(levels):
    '''Merge Laplacian pyramid'''

    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, K5X5 * 4, tf.shape(hi),
                    [1, 2, 2, 1]) + hi
    return img


def _normalize_std(img):
    '''Normalize image by making its standard deviation = 1.0'''

    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img), axis=(1, 2, 3),
                      keep_dims=True))
        return img / tf.maximum(std, config['EPS'])


def lap_normalize(img, scale_n):
    '''Perform the Laplacian pyramid normalization.'''

    tlevels = _lap_split_n(img, scale_n)
    tlevels = list(map(_normalize_std, tlevels))
    out = _lap_merge(tlevels)
    return out

################################################
# Class Visualization using deepdream approach #
################################################

def _visualization_by_layer_name(
    graph,
    value_feed_dict,
    input_tensor,
    layer_name,
    path_logdir,
    path_outdir,
    background_color,
    octaves,
    use_bilateral,
    use_tv_bregman
    ):
    """
....Generate and store filter visualization from the layer which has the name layer_name

....:param graph:
........TF graph
....:type graph_or_path: tf.Graph object

....:param value_feed_dict:
........Values of placeholders to feed while evaluting.
........dict : {placeholder1 : value1, ...}.
....:type value_feed_dict: dict or list

....:param input_tensor:
........Where to reconstruct
....:type input_tensor: tf.tensor object (Default = None)

....:param layer_name:
........Name of the layer to visualize
....:type layer_name: String

....:param path_logdir:
........<path-to-log-dir> to make log file for TensorBoard visualization
....:type path_logdir: String (Default = "./Log")

....:param path_outdir:
........<path-to-dir> to save results into disk as images
....:type path_outdir: String (Default = "./Output")

....:return:
........True if successful. False otherwise.
....:rtype: boolean
...."""

    start = -time.time()
    is_success = True

    # try:

    parsed_tensors = parse_tensors_dict(graph, layer_name,
            value_feed_dict)
    if parsed_tensors == None:
        return is_success
    (op_tensor, x, X_in, feed_dict) = parsed_tensors

    is_deep_dream = True
    with graph.as_default() as g:

        # computing reconstruction

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            X = X_in
            if input_tensor != None:
                X = get_tensor(graph=g, name=input_tensor.name)

            # original_images = sess.run(X, feed_dict = feed_dict)

            results = None

            # deepdream

            is_success = _deepdream(
                graph,
                sess,
                op_tensor,
                X,
                feed_dict,
                layer_name,
                path_outdir,
                path_logdir,
                background_color,
                octaves,
                use_bilateral,
                use_tv_bregman
                )
            is_deep_dream = False

            sess = None

    # except:
    # ....is_success = False
    # ....print("No Layer with layer name = %s" % (layer_name))
    # ....return is_success

    if is_deep_dream:
        is_success = _write_deepdream(results, layer_name, path_outdir,
                path_logdir)

    start += time.time()
    print 'Reconstruction Completed for %s layer. Time taken = %f s' \
        % (layer_name, start)

    return is_success


def _deepdream(
    graph,
    sess,
    op_tensor,
    X,
    feed_dict,
    layer,
    path_outdir,
    path_logdir,
    background_color,
    octaves,
    use_bilateral,
    use_tv_bregman
    ):

    tensor_shape = op_tensor.get_shape().as_list()

    with graph.as_default() as g:
        n = (config['N'] + 1) // 2
        feature_map = tf.placeholder(dtype=tf.int32)
        tmp1 = \
            tf.reduce_mean(tf.multiply(tf.gather(tf.transpose(op_tensor),
                           feature_map),
                           tf.diag(tf.ones_like(feature_map,
                           dtype=tf.float32))), axis=0)
        tmp2 = 1e-3 * tf.reduce_mean(tf.square(X), axis=(1, 2, 3))
        tmp = tmp1 - tmp2
        t_grad = tf.gradients(ys=tmp, xs=X)[0]

        lap_in = tf.placeholder(np.float32, name='lap_in')
        laplacian_pyramid = lap_normalize(lap_in,
                scale_n=config['NUM_LAPLACIAN_LEVEL'])

        image_to_resize = tf.placeholder(np.float32,
                name='image_to_resize')
        size_to_resize = tf.placeholder(np.int32, name='size_to_resize')
        resize_image = tf.image.resize_bilinear(image_to_resize,
                size_to_resize)

        with sess.as_default() as sess:
            tile_size = sess.run(tf.shape(X), feed_dict=feed_dict)[1:3]

            end = len(units)
            for k in range(0, end, n):
                c = n
                if k + n >= end:
                    c = end - end // n * n
                '''img = np.random.uniform(size=(c, tile_size[0],
                        tile_size[1], 3)) + 117.0'''
                background_color = np.float32([10, 90, 140])
                img = np.random.normal(background_color, 8, size=(c, tile_size[0], tile_size[1], 3))
                feed_dict[feature_map] = units[k:k + c]

                for (e, octave) in enumerate(octaves):
                    print 'octave {0} | layer {1} | scale {2} | iter_n {3}'.format(e,
                            layer, octave['scale'],
                            octave['iter_n'])

                    if len(octaves) > 0:
                        hw = np.float32(img.shape[1:3]) * octave['scale'
                                ]
                        img = sess.run(resize_image,
                                {image_to_resize: img,
                                size_to_resize: np.int32(hw)})
                        if use_tv_bregman:
                            for (i, im) in enumerate(img):
                                min_img = im.min()
                                max_img = im.max()
                                temp = denoise_tv_bregman((im - min_img)
                                        / (max_img - min_img),
                                        weight=config['TV_DENOISE_WEIGHT'])
                                img[i] = temp * (max_img - min_img) \
                                    + min_img
                    for j in range(octave['iter_n']):
                        sz = tile_size
                        (h, w) = img.shape[1:3]
                        sx = np.random.randint(sz[1], size=2)
                        sy = np.random.randint(sz[0], size=2)
                        img_shift = np.roll(np.roll(img, sx, 2), sy, 1)
                        grad = np.zeros_like(img)
                        sigma = octave['start_sigma'] + ((octave['end_sigma'] - octave['start_sigma']) * j) / octave['iter_n']
                        for y in range(0, max(h - sz[0] // 2, sz[0]),
                                sz[0] // 2):
                            for x in range(0, max(h - sz[1] // 2,
                                    sz[1]), sz[1] // 2):
                                feed_dict[X] = img_shift[:, y:y
                                        + sz[0], x:x + sz[1]]
                                try:
                                    grad[:, y:y + sz[0], x:x + sz[1]] = \
    sess.run(t_grad, feed_dict=feed_dict)
                                except:
                                    pass

                        lap_out = sess.run(laplacian_pyramid,
                                feed_dict={lap_in: np.roll(np.roll(grad,
                                -sx, 2), -sy, 1)})
                        img = img + lap_out
                        if use_bilateral:
                            for (i, im) in enumerate(img):
                                min_img = im.min()
                                max_img = im.max()
                                temp = denoise_bilateral((im - min_img)
                                        / (max_img - min_img), win_size=5, sigma_color=sigma, sigma_spatial=sigma, multichannel=True)
                                img[i] = temp * (max_img - min_img) \
                                    + min_img
                is_success = _write_deepdream(img, (layer, units, k),
                        path_outdir, path_logdir)
                print '%s -> featuremap completed.' \
                    % ', '.join(str(num) for num in units[k:k + c])
    return is_success


def class_visualization(
    graph_or_path,
    value_feed_dict,
    layer,
    classes,
    use_bilateral,
    use_tv_bregman,
    input_tensor=None,
    path_logdir='./Log',
    path_outdir='./Output',
    background_color = np.float32([117, 117, 117]),
    octaves=default_octaves
    ):

    is_success = True
    layers=layer
    if isinstance(layer, list):
        print 'Please only give classification layer name for reconstruction.'
        return False
    elif layer in dict_layer.keys():
        print 'Please only give classification layer name for reconstruction.'
        return False
    else:
        global units
        units = classes

        # is_success = _get_visualization(graph_or_path, value_feed_dict, input_tensor = input_tensor, layers = layer, method = "deepdream",
        #
        # ....path_logdir = path_logdir, path_outdir = path_outdir)

    if isinstance(graph_or_path, tf.Graph):
        PATH = _save_model(graph_or_path)
    elif isinstance(graph_or_path, string_types):
        PATH = graph_or_path
    else:
        print 'graph_or_path must be a object of graph or string.'
        is_success = False
        return is_success
    with tf.Graph().as_default() as g:
        sess = _graph_import_function(PATH)
        if not isinstance(layers, list):
            layers = [layers]

        for layer in layers:
            if layer != None and layer.lower() not in dict_layer.keys():
                is_success = _visualization_by_layer_name(
                    g,
                    value_feed_dict,
                    input_tensor,
                    layer,
                    path_logdir,
                    path_outdir,
                    background_color,
                    octaves,
                    use_bilateral,
                    use_tv_bregman
                    )
            elif layer != None and layer.lower() in dict_layer.keys():
                layer_type = dict_layer[layer.lower()]

                    # is_success = _visualization_by_layer_type(g, value_feed_dict, input_tensor, layer_type, method, path_logdir, path_outdir)

                net_layers = []
                for i in graph.get_operations():
                    if layer_type.lower() == i.type.lower():
                        net_layers.append(i.name)
                for net_layer in net_layers:
                    is_success = _visualization_by_layer_name(
                        g,
                        value_feed_dict,
                        input_tensor,
                        net_layer,
                        path_logdir,
                        path_outdir,
                        background_color,
                        octaves,
                        use_bilateral,
                        use_tv_bregman
                        )
            else:
                print 'Skipping %s . %s is not valid layer name or layer type' \
                    % (layer, layer)

    return is_success
