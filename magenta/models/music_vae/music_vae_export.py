from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow as tf
from collections import namedtuple
from tensorflow.python import saved_model
from magenta.models.music_vae import configs

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "Path to the checkpoint file")
flags.DEFINE_string("output_dir", None, "output directory for the saved model")
flags.DEFINE_string("config", None, "name of the VAE config to use")
flags.DEFINE_integer("batch_size", 10, "batch size the model is trained with")


# Function builds the VAE graph from a configuration object for exporting as a saved model
def build_vae_graph(vae_config, batch_size, **sample_kwargs):
    vae_graph = tf.Graph()
    with vae_graph.as_default():
        model = vae_config.model
        model.build(
            vae_config.hparams,
            vae_config.data_converter.output_depth,
            is_training=False)

        # Define nodes for the model inputs
        InputTensors = namedtuple(
            "VAEInputTensors",
            ["inputs", "controls", "inputs_length",
             "c_input", "z_input", "temperature", "max_length"]
        )
        temperature = tf.placeholder(tf.float32, shape=())

        if vae_config.hparams.z_size:
            z_input = tf.placeholder(tf.float32, shape=[batch_size, vae_config.hparams.z_size])
        else:
            z_input = None

        if vae_config.data_converter.control_depth > 0:
            c_input = tf.placeholder(tf.float32, shape=[None, vae_config.data_converter.control_depth])
        else:
            c_input = None

        inputs = tf.placeholder(
            tf.float32,
            shape=[batch_size, None, vae_config.data_converter.input_depth])
        controls = tf.placeholder(
            tf.float32,
            shape=[batch_size, None, vae_config.data_converter.control_depth])
        inputs_length = tf.placeholder(
            tf.int32,
            shape=[batch_size] + list(vae_config.data_converter.length_shape))

        max_length = tf.placeholder(tf.int32, shape=())
        input_tensors = InputTensors(inputs, controls, inputs_length, c_input, z_input, temperature, max_length)

        # used in the trained_model decode functions, run in the stored session
        OutputTensors = namedtuple(
            "VAEOutputTensors",
            ["outputs", "decoder_results", "mu", "sigma", "z"]
        )
        outputs, decoder_results = model.sample(
            batch_size,
            max_length=max_length,
            z=z_input,
            c_input=c_input,
            temperature=temperature,
            **sample_kwargs)

        if vae_config.hparams.z_size:
            q_z = model.encode(inputs, inputs_length, controls)
            mu = q_z.loc
            sigma = q_z.scale.diag
            z = q_z.sample()
        output_tensors = OutputTensors(outputs, decoder_results, mu, sigma, z)

        signature_defs = build_signature_defs(input_tensors, output_tensors)

    return vae_graph, signature_defs


# This function has too many arguments, pass a data structure later
def build_signature_defs(input_tensors, output_tensors):
    encode_signature_def = build_encode_signature(input_tensors, output_tensors)
    decode_signature_def = build_decode_signature(input_tensors, output_tensors)

    signature_def_map = {
        "encode": encode_signature_def,
        "decode": decode_signature_def
    }

    return signature_def_map


def build_encode_signature(input_tensors, output_tensors):
    # define the signature for the encoding operation
    encode_inputs = {
        "inputs": saved_model.utils.build_tensor_info(input_tensors.inputs),
        "controls": saved_model.utils.build_tensor_info(input_tensors.controls),
        "inputs_length": saved_model.utils.build_tensor_info(input_tensors.inputs_length)
    }
    encode_outputs = {
        "mu": saved_model.utils.build_tensor_info(output_tensors.mu),
        "sigma": saved_model.utils.build_tensor_info(output_tensors.sigma),
        "z": saved_model.utils.build_tensor_info(output_tensors.z)
    }

    encode_signature_def = saved_model.signature_def_utils.build_signature_def(
        inputs=encode_inputs,
        outputs=encode_outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    return encode_signature_def


def build_decode_signature(input_tensors, output_tensors):
    decode_inputs = {
        "temperature": saved_model.utils.build_tensor_info(input_tensors.temperature),
        "max_length": saved_model.utils.build_tensor_info(input_tensors.max_length),
        "z_input": saved_model.utils.build_tensor_info(input_tensors.z_input)
    }
    if input_tensors.c_input:
        decode_inputs["c_input"] = saved_model.utils.build_tensor_info(input_tensors.c_input)

    decode_outputs = {
        "outputs": saved_model.utils.build_tensor_info(output_tensors.outputs),
    }

    decode_signature_def = saved_model.signature_def_utils.build_signature_def(
        inputs=decode_inputs,
        outputs=decode_outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    return decode_signature_def


def export_saved_model(checkpoint_path, output_dir, config, batch_size, **sample_kwargs):
    graph, signature_defs = build_vae_graph(config, batch_size, **sample_kwargs)

    if tf.io.gfile.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        tf.logging.info("loading VAE checkpoint at: {}".format(checkpoint_path))
    elif not tf.io.gfile.exists(checkpoint_path + ".index"):
        raise ValueError("Invalid checkpoint path specified: {}".format(checkpoint_path))

    builder = saved_model.builder.SavedModelBuilder(output_dir)
    export_config_metadata(config, output_dir)

    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        session.run([tf.local_variables_initializer(), tf.tables_initializer()])
        saver.restore(session, checkpoint_path)
        builder.add_meta_graph_and_variables(
            session,
            tags=[saved_model.tag_constants.SERVING],
            signature_def_map=signature_defs
        )

    builder.save()


def export_config_metadata(config, output_dir):
    output_path = os.path.join(output_dir, "servable.metadata")
    with open(output_path, 'w') as outf:
        json.dump(config.serving_values(), outf)


if __name__ == '__main__':
    if FLAGS.config not in configs.CONFIG_MAP:
        raise ValueError('Invalid config name: %s' % FLAGS.config)
    config = configs.CONFIG_MAP[FLAGS.config]
    export_saved_model(FLAGS.checkpoint, FLAGS.output_dir, config, FLAGS.batch_size)
