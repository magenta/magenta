from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import tensorflow as tf
from tensorflow.python import saved_model


# Function builds the VAE graph from a configuration object for exporting as a saved model
def build_vae_graph(config, batch_size, **sample_kwargs):
    vae_graph = tf.Graph()
    with vae_graph.as_default():
        model = config.model
        model.build(
            config.hparams,
            config.data_converter.output_depth,
            is_training=False)

        # Define nodes for the model inputs
        temperature = tf.placeholder(tf.float32, shape=())

        if config.hparams.z_size:
            z_input = tf.placeholder(tf.float32, shape=[batch_size, config.hparams.z_size])
        else:
            z_input = None

        if config.data_converter.control_depth > 0:
            c_input = tf.placeholder(tf.float32, shape=[None, config.data_converter.control_depth])
        else:
            c_input = None

        inputs = tf.placeholder(
            tf.float32,
            shape=[batch_size, None, config.data_converter.input_depth])
        controls = tf.placeholder(
            tf.float32,
            shape=[batch_size, None, config.data_converter.control_depth])
        inputs_length = tf.placeholder(
            tf.int32,
            shape=[batch_size] + list(config.data_converter.length_shape))

        max_length = tf.placeholder(tf.int32, shape=())

        outputs, decoder_results = model.sample(
            batch_size,
            max_length=max_length,
            z=z_input,
            c_input=c_input,
            temperature=temperature,
            **sample_kwargs)

        if config.hparams.z_size:
            q_z = model.encode(inputs, inputs_length, controls)
            mu = q_z.loc
            sigma = q_z.scale.diag
            z = q_z.sample()

        signature_def = build_signature_def(inputs, controls, inputs_length, c_input, z_input, outputs)

    return vae_graph, signature_def


def build_signature_def(inputs, controls, inputs_length, c_input, z_input, outputs):
    # define the input signature for the VAE SignatureDef
    input_signature = {
        'inputs': saved_model.utils.build_tensor_info(inputs),
        'controls': saved_model.utils.build_tensor_info(controls),
        'inputs_length': saved_model.utils.build_tensor_info(inputs_length)
    }
    if c_input:
        input_signature['c_input'] = saved_model.utils.build_tensor_info(c_input)
    if z_input:
        input_signature['z_input'] = saved_model.utils.build_tensor_info(z_input)

    # define the output signature
    output_signature = {
        'outputs': saved_model.utils.build_tensor_info(outputs)
    }

    signature_def = saved_model.signature_def_utils.build_signature_def(
        inputs=input_signature,
        outputs=output_signature,
        method_name=saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    return signature_def


def export_saved_model(checkpoint_path, output_dir, config, batch_size, **sample_kwargs):
    graph, signature_def = build_vae_graph(config, batch_size, **sample_kwargs)

    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        tf.logging.info('loading VAE checkpoint at: {}'.format(checkpoint))
    elif not tf.gfile.Exists(checkpoint_path):
        raise ValueError('Invalid checkpoint path specified: {}'.format(checkpoint_path))

    builder = saved_model.builder.SavedModelBuilder(output_dir)

    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        session.run([tf.local_variables_initializer(), tf.tables_initializer()])
        saver.restore(session, checkpoint_path)
        builder.add_meta_graph_and_variables(
            session,
            tags=[saved_model.tag_constants.SERVING],
            signature_def_map={
                saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            },
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
        type=json.loads,
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
    export_saved_model(args.checkpoint, args.output_dir, args.config, args.batch_size)