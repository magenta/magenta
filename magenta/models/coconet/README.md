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

### References:

Uria, B., Murray, I., & Larochelle, H. (2014, January). A deep and tractable density estimator. In International Conference on Machine Learning (pp. 467-475).

Yao, L., Ozair, S., Cho, K., & Bengio, Y. (2014, September). On the equivalence between deep nade and generative stochastic networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 322-336). Springer, Berlin, Heidelberg.

## How to Use
We made template scripts for interacting with Coconet in four ways: training a
new model, generating from a model, evaluating a model, and evaluating
generated samples.  There are many variables in the script that are currently
set to defaults that may require customization.  Run these scripts from within the coconet
directory.

### Generating from Coconet

For generating from a pretrained model:

Download a model pretrained on J.S. Bach chorales from http://download.magenta.tensorflow.org/models/coconet/checkpoint.zip and pass the path up till the inner most directory as first argument to script.

sh sample_bazel.sh path_to_checkpoint

For example,
path_to_checkpoint could be $HOME/Downloads/checkpoint/coconet-64layers-128filters

### Training your own Coconet

sh train_bazel.sh

### Evaluating a trained Coconet

sh evalmodel_bazel.sh path_to_checkpoint

### Evaluating samples generated from Coconet

sh evalsample_bazel.sh path_to_checkpoint

