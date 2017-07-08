# Embedding_RNN

A neural network model to generate context conditioned outputs.

## Details

# Code

## Code Format

The code is structured as follows:

*   BUILD - specifies the build instructions for embedding_rnn
*   README.md - this document, outlines details and running instructions
*   embedding_rnn.py - main code, contains trianing/evaluation methods
*   embedding_rnn_test.py - basic testing skeleton for embedding_rnn development
*   run.py - generalized run code for running experiments over hyperparamters
*   script_utils.py - utility functions for running borg scripts
*   synthesize.py - processing code for synthesis, visualization and analysis

You can set the root_dir, log_root and data_dir in the borg file.

Experiment and job names should be unique and set in run.py. If they are not
unique then data saving will collide.

## Testing

When developing embedding_rnn you can quickly test the code by running:

~~~
bazel test --test_output=streamed //magenta/models/nsynth/embedding_rnn:embedding_rnn_test
~~~

## Training

### Train single model (local)

*   Uncomment default_run(local=True, exp_name='default_local') code in the main
    runtime of run.py
*   Provide the default_run method a unique exp_name
*   Model is saved to log_root set in embedding_rnn.py
*   Run embedding_rnn training:

~~~
python experimental/users/korymath/embedding_rnn/run.py
~~~

*   To run the model on borg, change the local flag to False
*   i.e. default_run(local=False, exp_name='default_borg')

## Generation

### Makes a unique directory to hold the generated synthesis and visuals.

Set the experiment name, model save path subdirectory and full path, and the
generation path to match the model trianed above.

~~~
exp_name = 'blog_13_unc0_nosam1_d1_pca1'
model_path_subdir = 'blog_13/'
gen_path_subdir = 'old_test_8_mse/'
model_save_path = ('korymath/embedding_rnn/' + model_path_subdir
                   + exp_name + '/')
~~~

### Match the hyperparameters and sweep names to those run in run.py

~~~
hps_model = embedding_rnn.default_hps()
hps_model.parse('rnn_size=%d' % 1024)
hps_model.parse('model=%s' % 'layer_norm')
hps_model.parse('num_layers=%d' % 2)
hps_model.parse('enc_rnn_size=%d' % 1024)
hps_model.parse('z_size=%d' % 128)
hps_model.parse('num_mixture=%d' % 20)
hps_model.parse('pca_dim_reduce=%d' % 1)
hps_model.parse('unconditional=%d' % 0)
hps_model.parse('difference_input=%d' % 1)
hps_model.parse('no_sample_vae=%d' % 1)
~~~
