from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import tensorflow as tf
from collections import namedtuple
from tensorflow.python import saved_model
from magenta.models.music_vae import configs


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
    sample_signature_def = build_sample_signature(input_tensors, output_tensors)
    encode_signature_def = build_encode_signature(input_tensors, output_tensors)
    decode_signature_def = build_decode_signature(input_tensors, output_tensors)

    signature_def_map = {
        'sample': sample_signature_def,
        'encode': encode_signature_def,
        'decode': decode_signature_def
    }

    return signature_def_map


# TODO create a named registry of signature def builders keyed by name for better type safety
# Sample seems almost exactly like decode from the trained model code, possibly redundant
def build_sample_signature(input_tensors, output_tensors):
    sample_inputs = {
        'temperature': saved_model.utils.build_tensor_info(input_tensors.temperature),
        'max_length': saved_model.utils.build_tensor_info(input_tensors.max_length)
    }
    if input_tensors.c_input is not None:
        sample_inputs['c_input'] = saved_model.utils.build_tensor_info(input_tensors.c_input)
    if input_tensors.z_input is not None:
        sample_inputs['z_input'] = saved_model.utils.build_tensor_info(input_tensors.z_input)

    sample_outputs = {
        'outputs': saved_model.utils.build_tensor_info(output_tensors.outputs)

    }

    sample_signature_def = saved_model.signature_def_utils.build_signature_def(
        inputs=sample_inputs,
        outputs=sample_outputs,
        method_name="sample"
    )

    return sample_signature_def


def build_encode_signature(input_tensors, output_tensors):
    # define the signature for the encoding operation
    encode_inputs = {
        'inputs': saved_model.utils.build_tensor_info(input_tensors.inputs),
        'controls': saved_model.utils.build_tensor_info(input_tensors.controls),
        'inputs_length': saved_model.utils.build_tensor_info(input_tensors.inputs_length)
    }
    encode_outputs = {
        'mu': saved_model.utils.build_tensor_info(output_tensors.mu),
        'sigma': saved_model.utils.build_tensor_info(output_tensors.sigma),
        'z': saved_model.utils.build_tensor_info(output_tensors.z)
    }

    encode_signature_def = saved_model.signature_def_utils.build_signature_def(
        inputs=encode_inputs,
        outputs=encode_outputs,
        method_name="encode"
    )

    return encode_signature_def


def build_decode_signature(input_tensors, output_tensors):
    decode_inputs = {
        'temperature': saved_model.utils.build_tensor_info(input_tensors.temperature),
        'max_length': saved_model.utils.build_tensor_info(input_tensors.max_length),
        'z_input': saved_model.utils.build_tensor_info(input_tensors.z_input)
    }
    if input_tensors.c_input:
        decode_inputs["c_input"] = saved_model.utils.build_tensor_info(input_tensors.c_input)

    # Question for magenta team about this, appears in trained model as a return value from decode,
    # but is not actually a tensor or operation
    decode_outputs = {
        'outputs': saved_model.utils.build_tensor_info(output_tensors.outputs),
        # 'decoder_results': saved_model.utils.build_tensor_info(decoder_results)

    }

    decode_signature_def = saved_model.signature_def_utils.build_signature_def(
        inputs=decode_inputs,
        outputs=decode_outputs,
        method_name='decode'
    )

    return decode_signature_def


def export_saved_model(checkpoint_path, output_dir, config, batch_size, **sample_kwargs):
    graph, signature_defs = build_vae_graph(config, batch_size, **sample_kwargs)

    if tf.io.gfile.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        tf.logging.info('loading VAE checkpoint at: {}'.format(checkpoint_path))
    elif not tf.io.gfile.exists(checkpoint_path + ".index"):
        raise ValueError('Invalid checkpoint path specified: {}'.format(checkpoint_path))

    builder = saved_model.builder.SavedModelBuilder(output_dir)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='output directory for the saved model'
    )
    parser.add_argument(
        '--config',
        required=True,
        default=None,
        help="""name of the VAE config to use """
    )
    parser.add_argument(
        '--batch-size',
        type=json.loads,
        help="""batch size the model was trained with"""
    )
    args = parser.parse_args()
    if args.config not in configs.CONFIG_MAP:
        raise ValueError('Invalid config name: %s' % args.config)
    config = configs.CONFIG_MAP[args.config]
    export_saved_model(args.checkpoint, args.output_dir, config, args.batch_size)
