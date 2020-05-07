## Coconet: Counterpoint by Convolution

Machine learning models of music typically break up the
task of composition into a chronological process, composing
a piece of music in a single pass from beginning to
end. On the contrary, human composers write music in
a nonlinear fashion, scribbling motifs here and there, often
revisiting choices previously made. In order to better
approximate this process, we train a convolutional neural
network to complete partial musical scores, and explore the
use of blocked Gibbs sampling as an analogue to rewriting.
Neither the model nor the generative procedure are tied to
a particular causal direction of composition.
Our model is an instance of orderless NADE (Uria 2014),
which allows more direct ancestral sampling. However,
we find that Gibbs sampling greatly improves sample quality,
which we demonstrate to be due to some conditional
distributions being poorly modeled. Moreover, we show
that even the cheap approximate blocked Gibbs procedure
from (Yao, 2014) yields better samples than ancestral sampling,
based on both log-likelihood and human evaluation.

Paper can be found at https://ismir2017.smcnus.org/wp-content/uploads/2017/10/187_Paper.pdf.
Huang, C. Z. A., Cooijmans, T., Roberts, A., Courville, A., & Eck, D. (2016). Counterpoint by Convolution. International Society of Music Information Retrieval (ISMIR).

## How to Use
There are four template scripts for interacting with Coconet in the following ways:
- 1) `sample_bazel.sh`, for generating samples from a pretrained model
- 2) `train_bazel.sh`, for training a new model
- 3) `evalmodel_bazel.sh`, for evaluating a model
- 4) `evalsample_bazel.sh`, for evaluating generated samples

There are many variables in the scripts that are currently set to defaults that may require customization. Run all these scripts from within the `coconet` directory.


### 1) Generating samples from a pre-trained model

The `sample_bazel.sh` script takes one argument, the path to the directory containing the checkpoint files named `checkpoint`.

**Usage:** `sh sample_bazel.sh <path_to_checkpoint_dir>`

First, you can download a model that has been pre-trained on J.S. Bach chorales from [here](http://download.magenta.tensorflow.org/models/coconet/checkpoint.zip). This download contains an inner directory named `coconet-64layers-128filters` that contains a handful of checkpoint files.

The file named `checkpoint` needs to point to the checkpoint binaries `best_model.ckpt` by containing the line `model_checkpoint_path: "best_model.ckpt"`. The file named `config` contains the hyperparameters and settings used to generate this model.

For example, depending on where you put the checkpoint download, you might run like this:
`sh sample_bazel.sh $HOME/Downloads/checkpoint/coconet-64layers-128filters`

### Training your own Coconet model

**Usage:** `sh train_bazel.sh`

This repo contains a test dataset in the `testdata` directory, but to train your own model you will want to download the full JSB Chorale dataset from [here](https://github.com/czhuang/JSB-Chorales-dataset). See the README in that repo for details on the dataset format. Specifically, you need the file named `Jsb16thSeparated.npz`.

(If you're getting a `ValueError: Cannot feed value of shape ... for Tensor which has shape ...`, you're likely using the wrong dataset.)

In `train_bazel.sh`, you will need to configure several directories:
- set `logdir` to the directory where you want to save experiment logs
- set `data_dir` to the directory where your training data lives
- set dataset to the class of your dataset, which will be `Jsb16thSeparated` if you used the above download.

### Evaluating a trained Coconet model

**Usage:** `sh evalmodel_bazel.sh <path_to_checkpoint_dir>``

### Evaluating samples generated from Coconet

**Usage:** `sh evalsample_bazel.sh <path_to_checkpoint_dir>`

## References:

Uria, B., Murray, I., & Larochelle, H. (2014, January). A deep and tractable density estimator. In International Conference on Machine Learning (pp. 467-475).

Yao, L., Ozair, S., Cho, K., & Bengio, Y. (2014, September). On the equivalence between deep nade and generative stochastic networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 322-336). Springer, Berlin, Heidelberg.
