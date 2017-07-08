#- HParams ---------------------------------------------------------------------
# hp_embedding_rnn = dict(
#     rnn_size=1024,
#     model="lstm",
#     num_layers=1,
#     enc_rnn_size=256,
#     z_size=64,
#     num_mixture=5
# )

# # blog post training parameters
# hp_embedding_rnn = dict(
#     rnn_size=512,
#     model="layer_norm",
#     num_layers=4,
#     enc_rnn_size=512,
#     z_size=128,
#     num_mixture=20
# )

# blog_11, 12
# blog post training parameters
# hp_embedding_rnn = dict(
#     rnn_size=1024,
#     model="layer_norm",
#     num_layers=1,
#     enc_rnn_size=512,  # 1024
#     z_size=128,
#     num_mixture=20
# )

# blog_13, low memory PCA
# blog post training parameters
# hp_embedding_rnn = dict(
#     rnn_size=1024,
#     model="layer_norm",
#     num_layers=2,
#     enc_rnn_size=1024,  # 1024
#     z_size=128,
#     num_mixture=20
# )

# blog_14, n_pca_components=2
# can IT BE DONE!
# hp_embedding_rnn = dict(
#     rnn_size=1024,
#     model="layer_norm",
#     num_layers=2,
#     enc_rnn_size=1024,  # 1024
#     z_size=128,
#     num_mixture=20,
#     n_pca_components=2)


def short_test(local=True):
  v = dict(exp_name="short_test")
  hp["num_steps"] = 100
  hp["save_every"] = 100
  hp["pca_dim_reduce"] = 1


def short_test_borg(local=False):
  v = dict(exp_name="short_test", job_name="embedding_rnn_borg_short")
  hp["num_steps"] = 1000
  hp["save_every"] = 100


def overfit(local=False):
  v = dict(exp_name="overfit_test")
  hp["overfit"] = 1


def sample_vae_sweep(local=False, run_all=False):
  unconditionals = [0, 1] if run_all else [0]
  job_name = "exp3"
  for unconditional in unconditionals:
    if unconditional == 1:
      no_sample_vaes = [0, 1] if run_all else [1]
      for no_sample_vae in no_sample_vaes:
        v = dict(
            job_name=job_name,
            exp_name="{}_unc_{}_no_s_{}".format(job_name, unconditional,
                                                no_sample_vae))
        hp["unconditional"] = unconditional
        hp["no_sample_vae"] = no_sample_vae
    else:
      v = dict(
          job_name=job_name,
          exp_name="{}_unc_{}".format(job_name, unconditional))
      hp["unconditional"] = unconditional


def z_size_sweep(local=False, run_all=False):
  z_sizes = [64, 128, 256, 512, 1024, 1984] if run_all else [64]
  for z_size in z_sizes:
    v = dict(exp_name="exp_zsize_{}".format(z_size))
    hp["z_size"] = z_size


def difference_test(local=False):
  difference_input = [0, 1]
  for diff in difference_input:
    v = dict(exp_name="exp_diff_input_{}".format(diff))
    hp["difference_input"] = diff


def small_sweep(local=False, run_all=False):
  rnn_size = [64, 128] if run_all else [64]
  for rnn_size in rnn_size:
    v = dict(exp_name="exp_rnn_size_{}".format(rnn_size))
    hp["rnn_size"] = rnn_size


def rnn_size_sweep(local=False, run_all=False):
  rnn_size = [256, 512, 1024] if run_all else [1024]
  for rnn_size in rnn_size:
    v = dict(exp_name="exp_rnn_size_{}".format(rnn_size))
    hp["rnn_size"] = rnn_size


def sweep_all_layer_norm(local=False, run_all=False):
  rnn_sizes = [128, 256, 512, 1024] if run_all else [512]
  models = ["layer_norm"]
  #   models = ["lstm", "layer_norm", "hyper"]
  num_layerss = [1, 2, 3, 4] if run_all else [1]
  enc_rnn_sizes = [128, 256, 512] if run_all else [512]
  z_sizes = [64, 128, 256, 512, 1024] if run_all else [64]
  num_mixtures = [5, 10, 20] if run_all else [5]

  for rnn_size in rnn_sizes:
    for model in models:
      for num_layers in num_layerss:
        for enc_rnn_size in enc_rnn_sizes:
          for z_size in z_sizes:
            for num_mixture in num_mixtures:
              v = dict(exp_name=("v9_rnn{}_{}_"
                                 "l{}_enc{}_z{}_mix{}").format(
                                     rnn_size, model, num_layers, enc_rnn_size,
                                     z_size, num_mixture))
              hp["rnn_size"] = rnn_size
              hp["model"] = model
              hp["num_layers"] = num_layers
              hp["enc_rnn_size"] = enc_rnn_size
              hp["z_size"] = z_size
              hp["num_mixture"] = num_mixture


def sweep_all_with_diff(local=False, run_all=False):
  """Sweetp model hyperparameters including differencing inputs."""
  rnn_sizes = [512, 1024] if run_all else [512]
  models = ["layer_norm"]
  #   models = ["lstm", "layer_norm", "hyper"]
  num_layerss = [1, 2] if run_all else [1]
  difference_inputs = [0, 1]
  enc_rnn_sizes = [512] if run_all else [512]
  z_sizes = [256, 512, 1024] if run_all else [64]
  num_mixtures = [5, 10, 20] if run_all else [5]

  for rnn_size in rnn_sizes:
    for model in models:
      for num_layers in num_layerss:
        for enc_rnn_size in enc_rnn_sizes:
          for z_size in z_sizes:
            for num_mixture in num_mixtures:
              for difference_input in difference_inputs:
                v = dict(exp_name=("v9_rnn{}_{}_"
                                   "l{}_enc{}_z{}_mix{}_d{}")
                         .format(rnn_size, model, num_layers, enc_rnn_size,
                                 z_size, num_mixture, difference_input))
                hp["rnn_size"] = rnn_size
                hp["model"] = model
                hp["num_layers"] = num_layers
                hp["enc_rnn_size"] = enc_rnn_size
                hp["difference_input"] = difference_input
                hp["z_size"] = z_size
                hp["num_mixture"] = num_mixture


def sweep_high_layer_norm(local=False, run_all=False):
  rnn_sizes = [1024, 1600, 2048] if run_all else [512]
  models = ["layer_norm"]
  #   models = ["lstm", "layer_norm", "hyper"]
  num_layerss = [1, 2, 3, 4] if run_all else [1]
  enc_rnn_sizes = [512, 1024, 2048] if run_all else [512]
  z_sizes = [512, 1024, 1600, 2048] if run_all else [64]
  num_mixtures = [20, 25, 30] if run_all else [5]

  for rnn_size in rnn_sizes:
    for model in models:
      for num_layers in num_layerss:
        for enc_rnn_size in enc_rnn_sizes:
          for z_size in z_sizes:
            for num_mixture in num_mixtures:
              v = dict(exp_name=("v8_rnn{}_{}_"
                                 "l{}_enc{}_z{}_mix{}").format(
                                     rnn_size, model, num_layers, enc_rnn_size,
                                     z_size, num_mixture))
              hp["rnn_size"] = rnn_size
              hp["model"] = model
              hp["num_layers"] = num_layers
              hp["enc_rnn_size"] = enc_rnn_size
              hp["z_size"] = z_size
              hp["num_mixture"] = num_mixture


def final_experiment(local=False, run_all=False):
  pca_dim_reduce = [0, 1] if run_all else [0]
  unconditionals = [0]  # [0, 1] if run_all else [0]
  difference_input = [0]  # [0, 1] if run_all else [0]
  job_name = "blog_14"
  for pca_red in pca_dim_reduce:
    for diff in difference_input:
      for unconditional in unconditionals:
        # if not unconditional, then condition on the autoencoder
        if unconditional == 0:
          # compare variational vs. basic autoencoder
          no_sample_vaes = [0, 1] if run_all else [1]
          for no_sample_vae in no_sample_vaes:
            v = dict(
                job_name=job_name,
                exp_name="{}_unc{}_nosam{}_d{}_pca{}".format(
                    job_name, unconditional, no_sample_vae, diff, pca_red))
            hp["pca_dim_reduce"] = pca_red
            hp["unconditional"] = unconditional
            hp["difference_input"] = diff
            hp["no_sample_vae"] = no_sample_vae
        else:
          v = dict(
              job_name=job_name,
              exp_name="{}_unc{}_d{}_pca{}".format(job_name, unconditional,
                                                   diff, pca_red))
          hp["pca_dim_reduce"] = pca_red
          hp["unconditional"] = unconditional
          hp["difference_input"] = diff


if __name__ == "__main__":
  ## Final Experiment
  final_experiment(local=False, run_all=True)

  ## Short local test
#   short_test(local=True)

#############################################################################
## Short borg test
# short_test_borg(local=False)

## Testing unconditional, no sampling of VAE
#   sample_vae_sweep(local=False, run_all=True)

## Run a sweep over embedding sizes
# z_size_sweep(local=False, run_all=True)

## Run a sweep over rnn sizes
# rnn_size_sweep(local=False run_all=True)

## Run a sweep over multiple parameters
# sweep_all(local=False, run_all=False)

## Run a sweep over multiple parameters with only layer_norm
# sweep_all_layer_norm(local=False, run_all=True)

## Run a sweep over multiple high parameters with only layer_norm
# sweep_high_layer_norm(local=False, run_all=True)

## Run a small sweep locally, local=True
# small_sweep(local=True, run_all=False)

## Run a small sweep on borg, local=False
# small_sweep(local=False, run_all=False)

## Run an overfitting test with a small training dataset
# overfit(local=False)

## Run a comparison between differencing inputs and not
#   difference_test(local=False)

## Run a sweep over multiple parameters including differencing
#   sweep_all_with_diff(local=False, run_all=True)

# default_run(local=True, exp_name='default_local')
