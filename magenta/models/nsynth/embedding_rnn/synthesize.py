"""Script for synthesizing embedding_rnn samples."""

import math
import operator as op
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.google as tf
import magenta.models.nsynth.wavenet.fastgen as fastgen
from magenta.models.nsynth.wavenet.embedding_rnn import embedding_rnn

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '/tmp/embedding_rnn/data',
                       'data directory numpy cpickle of vector file.')
tf.flags.DEFINE_string('log_root', '/tmp/embedding_rnn',
                       'directory to store all dumps.')
tf.flags.DEFINE_string('checkpoint_path', '',
                       'Path to embedding_rnn checkpoint to restore.')
tf.flags.DEFINE_string('nsynth_checkpoint_path', '',
                       'Path to NSynth checkpoint to restore.')
tf.flags.DEFINE_string('hparam', 'rnn_size=1000,data_set=train_z.npy',
                       'default model params string to be parsed by HParams.')


def invert_to_model_format(sample):
  # model the difference of reverse sequence
  # and add zero to also model initial dc.
  temp_zero_tensor = np.zeros([1, sample.shape[-1]])
  data = np.concatenate((sample, temp_zero_tensor), axis=0)
  data = data[::-1, :]
  return data[1:, :] - data[0:-1, :]


def revert_to_original_format(sample):
  # return data in original format
  # sample is in dimensions self.seq_length, self.seq_width
  return np.cumsum(sample, axis=0)[::-1]


def sample_embedding(sess,
                     s_model,
                     z=None,
                     pitches_sample=None,
                     temperature=1.0,
                     temperature_2=1.0,
                     greedy=False,
                     seq_len=125,
                     modelled_seq_width=16):
  """Sample the embeddings with the given model."""

  def get_pi_idx(x, pdf, greedy=False):
    """Get index of desired probability distribution to sample."""
    # greedy arg-max as in vector-rnn
    # makes it most deterministic
    if greedy:
      return np.argmax(pdf)
    # decide which index to get, network outputs distribution
    # makes sense to sample from it
    num_pdfs = pdf.size
    accumulate = 0
    for i in range(0, num_pdfs):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    print 'error with sampling ensemble'
    return -1

  # initialize width of generation and previous input (zeros) for LSTM
  # also initialize the output
  generation_width = modelled_seq_width
  prev_x = np.zeros((1, 1, generation_width))
  output = np.zeros((seq_len, generation_width), dtype=np.float32)

  # use random embedding if z is not provided, not used if unconditional
  if z is None:
    print 'No embedding inputted, generating one.'
    z = np.random.randn(s_model.hps.z_size)

  # generate previous state, use embedding and pitch if supplied
  if s_model.hps.unconditional:
    prev_state = sess.run(s_model.initial_state)
  else:
    prev_state = sess.run(
        s_model.initial_state,
        feed_dict={
            s_model.batch_z: [z],
            s_model.pitches_one_hot: pitches_sample
        })

  # for each set of the legth of the sequence
  for i in xrange(seq_len):
    # generate the feed for the run, no z or pitches if unconditional
    if s_model.hps.unconditional:
      feed = {s_model.input_x: prev_x, s_model.initial_state: prev_state}
    else:
      # otherwise the feed contains the embedding and pitch information
      # prev_state could be zero instead of the forward pass calculated above
      #
      feed = {
          s_model.input_x: prev_x,
          s_model.initial_state: prev_state,
          s_model.batch_z: [z],
          s_model.pitches_one_hot: pitches_sample
      }

    # run the model, return mixture parameters for Gaussian Mixture Model
    [logmix, mean, logstd, next_state] = sess.run([
        s_model.out_logmix, s_model.out_mean, s_model.out_logstd,
        s_model.final_state
    ], feed)

    # adjust temperatures
    # equation 7 from Sketch_RNN - https://arxiv.org/pdf/1704.03477.pdf
    # this is temp1 - logmix related temp
    logmix2 = np.copy(logmix) / temperature
    # numerical difficulties for float32,16 land
    # exponential can blow up, subtract max from numerator and denominator
    # maximum number is going to zero
    logmix2 -= logmix2.max()
    # standard softmax
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(generation_width, 1)

    # for each signal sample which mixture to sample
    chosen_mean = np.zeros(generation_width)
    chosen_logstd = np.zeros(generation_width)
    for j in range(generation_width):
      # which mixture should we choose
      idx = get_pi_idx(np.random.rand(), logmix2[j], greedy)
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    # sample a gaussian times the sqrt of the temperature.
    # based on unit analysis. variance as same units as temp (ref?)
    # temperature_1: std related temp, lower than temp1
    rand_gaussian = np.random.randn(generation_width) * math.sqrt(temperature_2)

    # next state is then defined by the chosen mean and std
    # if was just chosen_mean then this would be a regression
    # by giving a chosen_logstd give model flexibility to
    # choose standard deviation
    # compare the std to the mean
    next_x = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
    output[i, :] = next_x
    prev_x[0][0] = next_x
    prev_state = next_state

  return output


def build_reconstructions(sess,
                          eval_model=None,
                          sample_model=None,
                          num_rows=1,
                          data=None,
                          temperature=1.0,
                          temperature_2=1.0,
                          greedy=False,
                          modelled_seq_width=16):
  """Build the reconstruction set given a size and stored test embeddings."""
  seq_len = 125  # all sequences this length in NSynth
  input_samples = np.zeros((num_rows, seq_len, modelled_seq_width))
  input_pitches = np.zeros((num_rows, 128))
  output_results = np.zeros((num_rows, seq_len, modelled_seq_width))

  # perform reconstruction for each example
  for i in range(num_rows):
    # get the next sample from the test_reconstruction_set
    # raw data in original format
    sample = data[i, :, :]

    # set the pitch manually here
    # pitches are all set to 60 for these examples
    pitches = list(tf.one_hot(indices=60, depth=128).eval())

    # stack converted raw data samples in modelled format
    input_samples[i, :, :] = sample

    # stack
    input_pitches[i, :] = pitches

    # shape the modelled format embeddings and pitches for the feed
    temp_zero_padlet = np.zeros([1, 1, sample.shape[1]])

    # convert from the original format to modelled format if difference_input
    input_sample = np.concatenate((temp_zero_padlet, [sample]), axis=1)
    pitches_sample = np.reshape(pitches, [1, 128])

    # build the feed for the session run
    if eval_model.hps.unconditional:
      feed = {eval_model.input_data: input_sample}
    else:
      feed = {
          eval_model.input_data: input_sample,
          eval_model.pitches_one_hot: pitches_sample
      }

    # run the batch through the evaluation model to get back embeddings
    all_z = sess.run(eval_model.batch_z, feed_dict=feed)
    z = all_z[0]

    # sample from this embedding to get a resulting output
    result = sample_embedding(
        sess,
        sample_model,
        z,
        pitches_sample,
        temperature=temperature,
        temperature_2=temperature_2,
        greedy=greedy,
        seq_len=seq_len,
        modelled_seq_width=modelled_seq_width)
    output_results[i, :, :] = result

  return input_samples, input_pitches, output_results


def visualize_embed_interp(num_rows=1,
                           alphas=None,
                           z_orig=None,
                           z_recon=None,
                           test_recon_list_keys=None,
                           gen_path_root=None,
                           exp_name=None,
                           difference_input=1,
                           pca_dim_reduce=0,
                           pca=None):
  """Visualze embeddings and interpolations."""

  if sns:
    print 'Visualizing with Seaborn'

  # check if the generation directory exists, if not create it
  if not gfile.Exists(gen_path_root):
    print 'created generation directory:', gen_path_root
    gfile.MkDir(gen_path_root)

  # number of rows to generated interpolations
  rows = range(num_rows)

  # build interpolation smears
  z_mix = np.zeros((len(rows) * len(alphas), 125, 16))
  key_mix = []
  for j in range(len(rows)):
    row = rows[j]
    for i in range(len(alphas)):
      alpha = alphas[i]
      z_mix[j * len(alphas) + i, :, :] = (
          (1 - alpha) * z_recon[row, :, :] + alpha * z_orig[row, :, :])
      row_key = (test_recon_list_keys[row]
                 .replace(' ', '_').replace('[', '_').replace(']', '_'))
      key_mix.append('_mix_alpha_' + str(alpha) + '_row_' + str(row_key))

  # if you have sufficient rows then make the composite figure
  # make sure that the titles correspond to the slices rows if you sliced
  if num_rows >= 6:
    # build figure 1 comparing a variety of embeddings
    print 'Building composite figure.'
    fig, axes = plt.subplots(1, 6, figsize=(17, 4))

    axes[0].plot(z_orig[0, :, :])
    axes[0].set_title('Bass')
    axes[0].set_ylim([-10, 20])

    axes[1].plot(z_orig[1, :, :])
    axes[1].set_title('Fretless Bass')
    axes[1].set_ylim([-10, 20])

    axes[2].plot(z_orig[2, :, :])
    axes[2].set_title('Funky Piano')
    axes[2].set_ylim([-10, 20])

    axes[3].plot(z_orig[3, :, :])
    axes[3].set_title('Voice Alto')
    axes[3].set_ylim([-10, 20])

    axes[4].plot(z_orig[4, :, :])
    axes[4].set_title('Clavinet')
    axes[4].set_ylim([-10, 20])

    axes[5].plot(z_orig[5, :, :])
    axes[5].set_title('Groove Piano')
    axes[5].set_ylim([-10, 20])

    plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
    plt.subplots_adjust(top=0.85)

    # save fig1
    gen_file_path = (gen_path_root + exp_name + '_fig1a' + '.png')
    with gfile.GFile(gen_file_path, 'w') as f:
      plt.savefig(f)
    plt.close(fig)

    print 'Building composite reconstruction figure.'
    fig, axes = plt.subplots(1, 6, figsize=(17, 4))

    axes[0].plot(z_recon[0, :, :])
    axes[0].set_title('Bass')
    axes[0].set_ylim([-10, 20])

    axes[1].plot(z_recon[1, :, :])
    axes[1].set_title('Fretless Bass')
    axes[1].set_ylim([-10, 20])

    axes[2].plot(z_recon[2, :, :])
    axes[2].set_title('Funky Piano')
    axes[2].set_ylim([-10, 20])

    axes[3].plot(z_recon[3, :, :])
    axes[3].set_title('Voice Alto')
    axes[3].set_ylim([-10, 20])

    axes[4].plot(z_recon[4, :, :])
    axes[4].set_title('Clavinet')
    axes[4].set_ylim([-10, 20])

    axes[5].plot(z_recon[5, :, :])
    axes[5].set_title('Groove Piano')
    axes[5].set_ylim([-10, 20])

    plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
    plt.subplots_adjust(top=0.85)

    # save fig1
    gen_file_path = (gen_path_root + exp_name + '_fig1b' + '.png')
    with gfile.GFile(gen_file_path, 'w') as f:
      plt.savefig(f)
    plt.close(fig)

  # create composite figures
  # compare original and reconstruction
  for j in range(len(rows)):
    print 'row: ', j
    row = rows[j]
    # build figure 2
    if difference_input == 1:
      print 'Visualizing all, including modelled difference.'
      fig, axes = plt.subplots(1, 6, figsize=(17, 4))
      fig.suptitle(test_recon_list_keys[row], fontsize=16)

      axes[0].plot(z_orig[row, :, :])
      axes[0].set_title('raw data')
      axes[0].set_ylim([-10, 20])

      axes[1].plot(invert_to_model_format(z_orig[row, :, :]))
      axes[1].set_title('original - modelled')
      axes[1].set_ylim([-5, 5])

      axes[2].plot(invert_to_model_format(z_recon[row, :, :]))
      axes[2].set_title('recon - modelled')
      axes[2].set_ylim([-5, 5])

      axes[3].plot(z_orig[row, :, :])
      axes[3].set_title('original reverted')
      axes[3].set_ylim([-10, 20])

      axes[4].plot(z_recon[row, :, :])
      axes[4].set_title('reconstruction')
      axes[4].set_ylim([-10, 20])

      axes[5].plot((z_recon[row, :, :] - z_orig[row, :, :]))
      mse = np.mean((z_recon[row, :, :] - z_orig[row, :, :])**2)
      axes[5].set_title('diff, mse: %f' % mse)
      axes[5].set_ylim([-5, 5])

      plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
      plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
      plt.subplots_adjust(top=0.85)
    else:
      print 'Visualize only raw, reconstruction and residuals.'
      fig, axes = plt.subplots(1, 3, figsize=(8.5, 4))
      fig.suptitle(test_recon_list_keys[row], fontsize=16)

      axes[0].plot(z_orig[row, :, :])
      axes[0].set_title('raw data')
      axes[0].set_ylim([-10, 20])

      # print 'row', row
      # print z_recon[row, :, :]
      axes[1].plot(z_recon[row, :, :])
      axes[1].set_title('reconstruction')
      axes[1].set_ylim([-10, 20])

      axes[2].plot((z_recon[row, :, :] - z_orig[row, :, :]))
      mse = np.mean((z_recon[row, :, :] - z_orig[row, :, :])**2)
      axes[2].set_title('diff, mse: %f' % mse)
      axes[2].set_ylim([-5, 5])

      plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
      plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
      plt.subplots_adjust(top=0.85)

    # save fig2
    gen_file_path = (
        gen_path_root + exp_name + '_row' + str(j) + '_fig2' + '.png')
    with gfile.GFile(gen_file_path, 'w') as f:
      plt.savefig(f)
    plt.close(fig)

    # build figure three
    fig, axes = plt.subplots(2, len(alphas), figsize=(17, 6))
    fig.suptitle(test_recon_list_keys[row], fontsize=16)

    for i in range(len(alphas)):
      alpha = alphas[i]
      axes[0, i].plot((z_mix[j * len(alphas) + i, :, :]))
      axes[0, i].set_title('alpha:%s' % alpha)
      axes[0, i].set_ylim([-10, 20])

      axes[1, i].plot((z_mix[j * len(alphas) + i, :, :] - z_orig[row, :, :]))
      mse = np.mean((z_mix[j * len(alphas) + i, :, :] - z_orig[row, :, :])**2)
      axes[1, i].set_title('mse:%f' % mse)
      axes[1, i].set_ylim([-5, 5])

    plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
    plt.subplots_adjust(top=0.85)

    # save fig3
    gen_file_path = (
        gen_path_root + exp_name + '_row' + str(j) + '_fig3' + '.png')
    with gfile.GFile(gen_file_path, 'w') as f:
      plt.savefig(f)
    plt.close(fig)

    # save fig4 - fig4 is a composite of multiple models
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    fig.suptitle(exp_name, fontsize=16)

    # reconstruction in top plot
    # print 'row', row
    # print z_recon[row, :, :]
    axes[0].plot(z_recon[row, :, :])
    axes[0].set_title('reconstruction')
    axes[0].set_ylim([-10, 20])

    # residuals in the bottom plot
    axes[1].plot((z_recon[row, :, :] - z_orig[row, :, :]))
    mse = np.mean((z_recon[row, :, :] - z_orig[row, :, :])**2)
    axes[1].set_title('diff, mse: %f' % mse)
    axes[1].set_ylim([-5, 5])

    plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
    plt.subplots_adjust(top=0.85)

    # set the file name
    gen_file_path = (
        gen_path_root + exp_name + '_row' + str(j) + '_fig4' + '.png')
    with gfile.GFile(gen_file_path, 'w') as f:
      plt.savefig(f)
    plt.close(fig)

    # build a figure to visualize the PCA modelling and reconstruction
    if pca_dim_reduce == 1:
      print 'build PCA recon figure'
      fig, axes = plt.subplots(1, 6, figsize=(17, 4))
      fig.suptitle(test_recon_list_keys[row], fontsize=16)

      axes[0].plot(z_orig[row, :, :])
      axes[0].set_title('raw data')
      axes[0].set_ylim([-10, 20])

      # original PCA
      axes[1].plot(pca.transform(z_orig[row, :, :]))
      axes[1].set_title('original - PCA')
      axes[1].set_ylim([-5, 5])

      # recon PCA
      axes[2].plot(pca.transform(z_recon[row, :, :]))
      axes[2].set_title('recon - PCA')
      axes[2].set_ylim([-5, 5])

      axes[3].plot(z_orig[row, :, :])
      axes[3].set_title('original reverted')
      axes[3].set_ylim([-10, 20])

      axes[4].plot(z_recon[row, :, :])
      axes[4].set_title('reconstruction')
      axes[4].set_ylim([-10, 20])

      axes[5].plot((z_recon[row, :, :] - z_orig[row, :, :]))
      mse = np.mean((z_recon[row, :, :] - z_orig[row, :, :])**2)
      axes[5].set_title('diff, mse: %f' % mse)
      axes[5].set_ylim([-5, 5])

      plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
      plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
      plt.subplots_adjust(top=0.85)

      # set the file name
      gen_file_path = (
          gen_path_root + exp_name + '_row' + str(j) + '_fig5' + '.png')
      with gfile.GFile(gen_file_path, 'w') as f:
        plt.savefig(f)
      plt.close(fig)
    print 'Visuals saved to:', gen_path_root


def copy_hps(hps):
  temp = tf.HParams()
  temp._init_from_proto(hps.to_proto())
  return temp


def run_synth_and_viz(hps_model=None,
                      model_save_path=None,
                      num_rows=1,
                      gen_path_subdir='blog_post_large/',
                      just_figures=0,
                      exp_name=None,
                      temperature=1.0,
                      temperature_2=1.0,
                      greedy=True,
                      slice_test_set=None):
  """Synthesize and visualize."""

  print 'loading data files, copy model hyperparameters'
  [_, hps_model, eval_hps_model, sample_hps_model] = embedding_rnn.load_dataset(
      hps_model, load_data=False)

  print 'reset graph'
  embedding_rnn.reset_graph()

  print 'build model'
  model = embedding_rnn.Model(hps_model)
  print 'model built', model

  print 'build evaluation model'
  # copy hyperparameters from hps_model
  eval_hps_model = copy_hps(hps_model)
  eval_hps_model.parse('use_input_dropout=%d' % 0)
  eval_hps_model.parse('use_recurrent_dropout=%d' % 0)
  eval_hps_model.parse('use_output_dropout=%d' % 0)
  eval_hps_model.parse('is_training=%d' % 0)
  eval_hps_model.parse('batch_size=%d' % 1)
  # print eval_hps_model
  eval_model = embedding_rnn.Model(eval_hps_model, reuse=True)
  print 'evaluation model built', eval_model

  print 'build sampling model'
  # copy hyperparameters from hps_model
  sample_hps_model = copy_hps(eval_hps_model)
  sample_hps_model.parse('max_seq_len=%d' % 1)
  # print sample_hps_model
  sample_model = embedding_rnn.Model(sample_hps_model, reuse=True)
  print 'sample model built', sample_model

  print 'start the session'
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  print 'load trained model', model_save_path
  # load model
  embedding_rnn.load_model(sess, model_save_path)

  # load testing reconstruction set
  with gfile.Open(('60_key.npy'), 'r') as f:
    test_recon_list_keys = np.load(f)
  print test_recon_list_keys[:1]
  with gfile.Open(('/c60_z.npy'), 'r') as f:
    test_recon_z = np.load(f)
  print 'test_recon_z.shape', test_recon_z.shape

  # only process certain row indecies if slice_test_set is set
  if slice_test_set is not None:
    print 'slicing a test set', slice_test_set
    test_recon_list_keys = test_recon_list_keys[slice_test_set]
    test_recon_z = test_recon_z[slice_test_set]

  # set some default parameters regardless of PCA dimensionality reduction
  # sequence width
  seq_width = 16

  # model the difference of reverse sequence
  # and add zero to also model initial dc.
  if hps_model.difference_input:
    # allocate an array, Number of Samples x 1 x 16
    temp_zero_tensor = np.zeros([test_recon_z.shape[0], 1, seq_width])
    # concatenate the raw data with single empty dimension along axis=1
    data = np.concatenate((test_recon_z, temp_zero_tensor), axis=1)
    # reverse the data along axis=1
    data = data[:, ::-1, :]
    # perform a difference along the second dimension for all samples
    # and over all the channels, now all the data will start at 0 and
    # be the correct length
    data = data[:, 1:, :] - data[:, 0:-1, :]
  else:
    # model the raw data with no difference transform
    data = test_recon_z

  # Build Reconstruction
  # Sampling reconstruction parameters
  temperature = temperature  # 0.5
  temperature_2 = temperature_2  # 0.3
  greedy = greedy  # True

  # modelled sequence may change if dimensionality reduction is used
  modelled_seq_width = seq_width
  pca = None

  if hps_model.pca_dim_reduce == 1:
    # If working with PCA data in the model then need to transform the
    # test recon z with the loaded pickle for PCA
    pkl_file = gfile.Open(model_save_path + 'pca.pkl', 'r')
    pca = pickle.load(pkl_file)
    modelled_seq_width = pca.n_components
    print pca

    # reduce dimensionality
    data = np.reshape(
        pca.transform(
            np.reshape(data, (data.shape[0] * hps_model.max_seq_len, seq_width
                             ))), (data.shape[0], hps_model.max_seq_len,
                                   modelled_seq_width))

  # build reconstructions
  print 'Building reconstruction, difference_input:', hps_model.difference_input
  z_orig, _, z_recon = build_reconstructions(
      sess=sess,
      eval_model=eval_model,
      sample_model=sample_model,
      num_rows=num_rows,
      data=data,
      temperature=temperature,
      temperature_2=temperature_2,
      greedy=greedy,
      modelled_seq_width=modelled_seq_width)

  if hps_model.pca_dim_reduce == 1:
    # if modelling the PCA then need to load the PCA model and reconstruct the
    # original dimensional data z_recon is (n_samples, 125, n_pca_components)
    # need (n_samples, 125, 16) for what comes next.

    # transform and reshape the reconstructions
    z_recon = np.reshape(
        pca.inverse_transform(
            np.reshape(z_recon, (z_recon.shape[0] * hps_model.max_seq_len,
                                 modelled_seq_width))),
        (z_recon.shape[0], hps_model.max_seq_len, seq_width))

    # transform and reshape the original input samples
    z_orig = np.reshape(
        pca.inverse_transform(
            np.reshape(z_orig, (z_orig.shape[0] * hps_model.max_seq_len,
                                modelled_seq_width))),
        (z_orig.shape[0], hps_model.max_seq_len, seq_width))

  # we are now working with the full size embeddings
  # each embedding is thus max_seq_len x seq_width

  # need to revert to the original format
  # for pca and diff need to follow the correct order
  # load, diff, pca, model/recon, pca, diff, vizualize
  if hps_model.difference_input:
    print 'revert original and reconstruction to original format'
    for i in range(num_rows):
      z_orig[i, :, :] = revert_to_original_format(z_orig[i, :, :])
      z_recon[i, :, :] = revert_to_original_format(z_recon[i, :, :])
  print 'reconstructions built'

  # calculate the MSE between all the originals and all the reconstructions for
  # the given experimental condition
  print 'calculating MSE for all reconstructions'
  mse = np.mean(np.mean((z_orig - z_recon)**2, axis=2), axis=1)
  # print mse
  # print mse.shape

  print 'build interpolations'
  # number of samples to generate interpolations over
  num_rows = num_rows
  rows = range(num_rows)

  # range of interolation smears
  # nonlinear interpolation to emphasize smears
  alphas = [0.2, 0.5, 0.8, 0.85, 0.9, 0.95]  # np.linspace(0.05, 0.95, num=5)

  # placeholders for the smears
  z_mix = np.zeros((len(rows) * len(alphas), 125, 16))
  key_mix = []

  for j in range(len(rows)):
    for i in range(len(alphas)):
      alpha = alphas[i]
      z_mix[j * len(alphas) + i, :, :] = (
          (1 - alpha) * z_recon[j, :, :] + alpha * z_orig[j, :, :])
      row_key = (test_recon_list_keys[j].replace(' ', '_').replace('[', '_')
                 .replace(']', '_'))
      key_mix.append('_mix_alpha_' + str(alpha) + '_row_' + str(row_key))

  z_all = np.vstack([z_orig, z_recon, z_mix])

  # need to ensure that the weird characters are removed from the strings
  # also only use the number of keys that are used in the recon set
  key_all = np.vstack([[
      'orig_{}'.format(
          key.replace(' ', '_').replace('[', '_').replace(']', '_'))
      for key in test_recon_list_keys[:num_rows]
  ], [
      'recon_{}'.format(
          key.replace(' ', '_').replace('[', '_').replace(']', '_'))
      for key in test_recon_list_keys[:num_rows]
  ]]).flatten()

  # add the keys for the mixing parameters
  key_all = np.append(key_all, key_mix)

  # test to make sure that original and reconstructions are saved
  # these should output original and reconstruction keys
  print 'num_rows:', num_rows
  print 'z_all.shape', z_all.shape
  print 'generating file paths'
  gen_path_root = ('fastgen-blog/' + gen_path_subdir)

  ## Build and save figures
  visualize_embed_interp(
      num_rows=num_rows,
      alphas=alphas,
      z_orig=z_orig,
      z_recon=z_recon,
      test_recon_list_keys=test_recon_list_keys,
      gen_path_root=gen_path_root,
      exp_name=exp_name,
      difference_input=hps_model.difference_input,
      pca_dim_reduce=hps_model.pca_dim_reduce,
      pca=pca)

  # perform the generation for all the z in z_all with matching
  # keys in key_all
  if just_figures == 0:
    gen_file_path = []
    gen_files = []
    for key in key_all:
      gen_file_path = gen_path_root + exp_name + key + '.wav'
      gen_files.append(gfile.GFile(gen_file_path, 'w'))
    # print gen_files

    ## Synthesize the originals, interpolations, and reconstructed embeddings
    # Define the location of the NSynth WaveNet model checkpoint

    # test generation with the noise model
    checkpoint_path = ('')

    print 'Using checkpoint:', checkpoint_path
    # Synthesize audio with fastgen.synthesize() from third_party magenta
    print 'Synthesizing samples'
    fastgen.synthesize(
        encodings=z_all, save_paths=gen_files, checkpoint_path=checkpoint_path)
    print 'Samples synthesized'

    # make sure that the files are appropriately closed
    print 'Closing files'
    for f in gen_files:
      f.close()
    print 'Files closed.'

    print 'Done, returning MSE for each reconstruction'
  return mse, gen_path_root


def make_aggregate_boxplot(all_mse=None,
                           gen_path_root=None,
                           num_rows=None,
                           temperature=None,
                           temperature_2=None,
                           greedy=None):
  """Make aggregat box and swarm plot with the dictionary of errors."""
  # set aggregate plot data
  plot_data = all_mse

  # print the minimum for each experiment
  for k, v in plot_data.iteritems():
    print 'minimum for: ', k
    min_idx = min(enumerate(v), key=op.itemgetter(1))[0]
    print 'row{}, mse: {}'.format(min_idx, v[min_idx])

  # sort keys, medians, and arrays of mse together
  # sort the box plot by the median of the MSE
  (sorted_keys, _, sorted_vals) = zip(*sorted(
      [(k, np.median(v), v) for k, v in plot_data.items()], key=lambda x: x[1]))

  fig, axes = plt.subplots(1, 1, figsize=(17, 8))
  title = 'N{}_t{}_t2{}_g{}'.format(num_rows, temperature, temperature_2,
                                    greedy)
  fig.suptitle(title, fontsize=16)

  # build box and swarm plot
  sns.axlabel(xlabel='', ylabel='MSE', fontsize=16)
  sns.boxplot(data=sorted_vals, width=.4, ax=axes)

  plt.xticks(plt.xticks()[0], sorted_keys)
  # remove the experiment prefix
  _ = [tick.label.set_fontsize(16) for tick in axes.yaxis.get_major_ticks()]
  xlabels = [item.get_text()[8:] for item in axes.get_xticklabels()]
  axes.set_xticklabels(xlabels, rotation=90, fontsize=16)
  axes.set(yscale='log')
  plt.subplots_adjust(top=0.85)
  plt.tight_layout()

  # set the file name
  gen_file_path = (gen_path_root + 'aggregate_fig6' + '.png')
  with gfile.GFile(gen_file_path, 'w') as f:
    plt.savefig(f)
  plt.close(fig)
  print 'Fig6 saved to:', gen_path_root


def blog_5():
  """Synthesis code to match final_experiment in run.py."""
  # default hyperparameters
  hps_model = embedding_rnn.default_hps()

  ## Set training specific hyperparameters
  # NOTE: these parameters MUST match the loaded model hyperparameters
  # blog_post_large default parameters
  # as defined in script_utils.py
  hps_model.parse('rnn_size=%d' % 512)
  hps_model.parse('model=%s' % 'layer_norm')
  hps_model.parse('num_layers=%d' % 4)
  hps_model.parse('enc_rnn_size=%d' % 512)
  hps_model.parse('z_size=%d' % 128)
  hps_model.parse('num_mixture=%d' % 20)

  # generation job/synthesize/visualize parameters
  model_path_subdir = 'blog_5/'
  gen_path_subdir = 'blog_5_full_gen/'
  job_name = 'blog_5'
  num_rows = 1
  just_figures = 1
  temperature = 0.5
  temperature_2 = 0.3
  greedy = False

  print 'temps and greedy', temperature, temperature_2, greedy

  ## sweep parameters:
  unconditionals = [0, 1]
  difference_input = [0, 1]
  no_sample_vaes = [0, 1]

  # final experiment
  # run the sweep
  for diff in difference_input:
    for unconditional in unconditionals:
      # if not unconditional, then condition on the autoencoder
      if unconditional == 0:
        # compare variational vs. basic autoencoder
        for no_sample_vae in no_sample_vaes:
          exp_name = '{}_unc{}_nosam{}_d{}'.format(job_name, unconditional,
                                                   no_sample_vae, diff)
          hps_model.parse('unconditional=%d' % unconditional)
          hps_model.parse('difference_input=%d' % diff)
          hps_model.parse('no_sample_vae=%d' % no_sample_vae)
          model_save_path = (
              '/cns/is-d/home/brain-arts/rs=6.3/experiments/'
              'korymath/embedding_rnn/' + model_path_subdir + exp_name + '/')
          # run the synthesis and visualization
          run_synth_and_viz(
              hps_model=hps_model,
              model_save_path=model_save_path,
              gen_path_subdir=gen_path_subdir,
              num_rows=num_rows,
              just_figures=just_figures,
              exp_name=exp_name,
              temperature=temperature,
              temperature_2=temperature_2,
              greedy=greedy)
      else:
        exp_name = '{}_unc{}_d{}'.format(job_name, unconditional, diff)
        hps_model.parse('unconditional=%d' % unconditional)
        hps_model.parse('difference_input=%d' % diff)
        model_save_path = (
            '/cns/is-d/home/brain-arts/rs=6.3/experiments/'
            'korymath/embedding_rnn/' + model_path_subdir + exp_name + '/')
        # run the synthesis and visualization
        run_synth_and_viz(
            hps_model=hps_model,
            model_save_path=model_save_path,
            gen_path_subdir=gen_path_subdir,
            num_rows=num_rows,
            just_figures=just_figures,
            exp_name=exp_name,
            temperature=temperature,
            temperature_2=temperature_2,
            greedy=greedy)


def final_experiment():
  """Synthesis code to match final_experiment in run.py."""
  # default hyperparameters
  hps_model = embedding_rnn.default_hps()

  ## Set training specific hyperparameters
  # NOTE: these parameters MUST match the loaded model hyperparameters
  # blog_post_large default parameters
  # as defined in script_utils.py

  # generation job/synthesize/visualize parameters
  job_name = 'blog_13'
  gen_path_subdir = 'blog_13_greedy_06_03_agg_noise_borg/'
  slice_test_set = range(912)
  num_rows = 4  # len(slice_test_set)  # 912
  # check slice test set and number of rows
  if num_rows > len(slice_test_set):
    # num_rows must be < slice_test_set,
    # else set num_rows to len(slice_test_set)
    num_rows = len(slice_test_set)
  just_figures = 0
  temperature = 0.6
  temperature_2 = 0.3
  greedy = False

  model_path_subdir = 'blog_13/'
  hps_model.parse('rnn_size=%d' % 1024)
  hps_model.parse('model=%s' % 'layer_norm')
  hps_model.parse('num_layers=%d' % 2)
  hps_model.parse('enc_rnn_size=%d' % 1024)
  hps_model.parse('z_size=%d' % 128)
  hps_model.parse('num_mixture=%d' % 20)

  ## sweep parameters:
  pca_dim_reduce = [0]  # [0, 1]
  unconditionals = [0]  # [0, 1]
  difference_input = [0]  # [0, 1]

  # collect all mse in a dictionary
  # each experiment is keyed by experiment name, and contains array of errors
  all_mse = {}
  all_mse_counter = 0

  # final experiment
  # run the sweep
  for pca_red in pca_dim_reduce:
    for diff in difference_input:
      for unconditional in unconditionals:
        # if not unconditional, then condition on the autoencoder
        if unconditional == 0:
          # compare variational vs. basic autoencoder
          no_sample_vaes = [0]  # [0, 1]
          for no_sample_vae in no_sample_vaes:
            exp_name = '{}_unc{}_nosam{}_d{}_pca{}'.format(
                job_name, unconditional, no_sample_vae, diff, pca_red)
            hps_model.parse('pca_dim_reduce=%d' % pca_red)
            hps_model.parse('unconditional=%d' % unconditional)
            hps_model.parse('difference_input=%d' % diff)
            hps_model.parse('no_sample_vae=%d' % no_sample_vae)
            model_save_path = (
                '/cns/is-d/home/brain-arts/rs=6.3/experiments/'
                'korymath/embedding_rnn/' + model_path_subdir + exp_name + '/')
            # run the synthesis and visualization
            (all_mse[exp_name], gen_path_root) = run_synth_and_viz(
                hps_model=hps_model,
                model_save_path=model_save_path,
                gen_path_subdir=gen_path_subdir,
                num_rows=num_rows,
                just_figures=just_figures,
                exp_name=exp_name,
                temperature=temperature,
                temperature_2=temperature_2,
                greedy=greedy,
                slice_test_set=slice_test_set)

            # print best row for this experiment
            print 'minimum for: ', exp_name
            min_idx = min(enumerate(all_mse[exp_name]), key=op.itemgetter(1))[0]
            print 'row{}, mse: {}'.format(min_idx, all_mse[exp_name][min_idx])

            # save all_mse on each step
            print 'updating saved error dictionary'
            pickle_path = (
                gen_path_root + 'all_mse{}.pkl'.format(all_mse_counter))
            all_mse_pkl = gfile.Open(pickle_path, 'a+')
            pickle.dump(all_mse, all_mse_pkl)
            print 'saved to: ' + pickle_path
            all_mse_counter += 1
            all_mse_pkl.close()

        else:
          exp_name = '{}_unc{}_d{}_pca{}'.format(job_name, unconditional, diff,
                                                 pca_red)
          hps_model.parse('pca_dim_reduce=%d' % pca_red)
          hps_model.parse('unconditional=%d' % unconditional)
          hps_model.parse('difference_input=%d' % diff)
          model_save_path = (
              '/cns/is-d/home/brain-arts/rs=6.3/experiments/'
              'korymath/embedding_rnn/' + model_path_subdir + exp_name + '/')
          # run the synthesis and visualization
          (all_mse[exp_name], gen_path_root) = run_synth_and_viz(
              hps_model=hps_model,
              model_save_path=model_save_path,
              gen_path_subdir=gen_path_subdir,
              num_rows=num_rows,
              just_figures=just_figures,
              exp_name=exp_name,
              temperature=temperature,
              temperature_2=temperature_2,
              greedy=greedy,
              slice_test_set=slice_test_set)

          # print best row for this experiment
          print 'minimum for: ', exp_name
          min_idx = min(enumerate(all_mse[exp_name]), key=op.itemgetter(1))[0]
          print 'row{}, mse: {}'.format(min_idx, all_mse[exp_name][min_idx])

          # save all_mse on each step
          print 'updating saved error dictionary'
          pickle_path = (
              gen_path_root + 'all_mse{}.pkl'.format(all_mse_counter))
          all_mse_pkl = gfile.Open(pickle_path, 'a+')
          pickle.dump(all_mse, all_mse_pkl)
          print 'saved to: ' + pickle_path
          all_mse_counter += 1
          all_mse_pkl.close()

  # make aggregate plot for figure 6
  make_aggregate_boxplot(
      all_mse=all_mse,
      gen_path_root=gen_path_root,
      num_rows=num_rows,
      temperature=temperature,
      temperature_2=temperature_2,
      greedy=greedy)
  print 'done final experiment synthesis and visualization'


def single_borg_test():
  """Synthesize and visualize from a single model."""
  # default hyperparameters
  hps_model = embedding_rnn.default_hps()

  ## Set training specific hyperparameters
  # NOTE: these parameters MUST match the loaded model hyperparameters
  # blog_post_large default parameters
  # as defined in script_utils.py
  exp_name = 'blog_13_unc0_nosam1_d1_pca1'
  model_path_subdir = 'blog_13/'
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

  # generation parameters
  gen_path_subdir = 'old_test_8_mse/'
  slice_test_set = [10, 11, 35, 64, 100, 121]
  # TODO(korymath): fix slice test set
  # num_rows must be < slice_test_set, else set num_rows to len(slice_test_set)
  num_rows = 6
  just_figures = 1
  temperature = 0.5
  temperature_2 = 0.3
  greedy = True

  # collect all mse in a dictionary
  all_mse = {}

  model_save_path = (
      'korymath/embedding_rnn/' + model_path_subdir + exp_name + '/')

  # return the dictionary of errors, and the generation file path
  (all_mse[exp_name], gen_path_root) = run_synth_and_viz(
      hps_model=hps_model,
      model_save_path=model_save_path,
      gen_path_subdir=gen_path_subdir,
      num_rows=num_rows,
      just_figures=just_figures,
      exp_name=exp_name,
      temperature=temperature,
      temperature_2=temperature_2,
      greedy=greedy,
      slice_test_set=slice_test_set)

  # make aggregate plot for figure 6
  make_aggregate_boxplot(
      all_mse=all_mse,
      gen_path_root=gen_path_root,
      num_rows=num_rows,
      temperature=temperature,
      temperature_2=temperature_2,
      greedy=greedy)


def main(_):
  print 'embedding_rnn synthesis'
  print 'hyper params'

  # run the final experiment
  # need to make sure that generation folder exists
  final_experiment()

  # run full gen on blog_5 models
  # blog_5()

  # run a single model test
  # single_borg_test()


if __name__ == '__main__':
  tf.app.run(main)
