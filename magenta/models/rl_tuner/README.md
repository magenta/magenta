# Tuning RNNs with RL

This code implements the models described in [this research paper][our arxiv],
and [this blog][blog post]. The idea is to take an LSTM that has been trained
to predict the next note in a monophonic melody &mdash; called a Note RNN
&mdash; and enhance it using reinforcement learning (RL).

The RLTuner class implements a [Deep Q Network (DQN)][dqn], in which the Q
network  learns the reward value of taking actions (playing notes) given the
state of the environment (the melody composed so far). The reward that the
network learns comes from two sources: 1) a set of music theory reward
functions, and 2) the output of a trained Note RNN, which gives *p(a|s)*, the
probability of playing the next note *a* given the state of the composition *s*,
as originally learned from data. This combination allows the model to maintain
what it learned from data, while constraining it to conform to a set of music
theory rules.

Using a checkpoint file storing a trained Note RNN, the NoteRNNLoader class is
used to load three copies of the Note RNN into RLTuner. Two copies supply the
initial values for the Q-network and Target-Q-network in the DQN algorithm,
while the third is used as a Reward RNN, which supplies the p(a|s) values in the
reward function. Note that the Reward RNN remains fixed; its weights are not
updated during training, so it always represents the note probabilities learned
from data.

The music theory reward functions are designed to constrain the actions of the
network so that it chooses notes in accordance with a musical structure; for
example, choosing harmonious interval steps and playing notes within the same
key. Several reward functions have been written, but these could easily be
improved and extended!

In addition to the normal Q function, this code provides the ability to train
the network with the [Psi learning][psi learning] and [G learning][g learning]
functions, which can be set with the `algorithm` hyperparameter. For details
on each algorithm, see [our paper][our arxiv].

## Code structure
*   In the constructor, RLTuner loads the `q_network`, `target_q_network`, and
    `reward_rnn` from a checkpointed Note RNN.

*   The tensorflow graph architecture is defined in the `build_graph`
    function.

*   The model is trained using the `train` function. It will continuously
    place notes by calling `action`, receive rewards using `collect_reward`,
    and save these experiences using `store`.

*   The network weights are updated using `training_step`, which samples
    minibatches of experience from the model's `experience` buffer and uses
    this to compute gradients based on the loss function in `build_graph`.

*   During training, the function `evaluate_model` is occasionally run to
    test how much reward the model receives from both the Reward RNN and the
    music theory functions.

*   After the model is trained, you can use the `save_model_and_figs` function
    to save a checkpoint of the model and a set of figures of the rewards over
    time.

*   Finally, use `generate_music_sequence` to generate a melody with your
    trained model! You can also call this function before training, to see how
    the model's songs have improved with training! If you set the
    `visualize_probs` parameter to *True*, it will also plot the
    note probabilities of the model over time.

## Running the code
To start using the model, first set up your [Magenta
environment](/README.md).
you can either use a pre-trained model or train your own.

To train the model you can use the jupyter notebook
[RL_Tuner.ipynb](https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/RL_Tuner.ipynb) found
in our [Magenta Demos](https://github.com/tensorflow/magenta-demos) repository or you can simply run:

```
rl_tuner_train
```

## Tuning your own model

By default, if you don't provide a Note RNN checkpoint file to load, the code
will automatically download and use the checkpointed model we used for
[our paper][our arxiv] from [here][note rnn ckpt].

If you want to use your own model, you need to pass in the directory containing
it using the `note_rnn_checkpoint_dir`, and the hyperparameters you used to
train it via `note_rnn_hparams`. You can also pass in a path to the checkpoint
file directly using `note_rnn_checkpoint_file`.

We also support tuning a *basic_rnn* trained using the Magenta code! To tune
a basic_rnn, use the same `note_rnn_checkpoint_dir` parameter, but set the
`note_rnn_type` parameter to 'basic_rnn'. We also provide the script
`unpack_bundle` (in magenta/scripts) to help you extract a checkpoint file from
one of the [pre-trained magenta bundles][magenta pretrained].

## Improving the model
If you have ideas for improving the sound of the model based on your own rules
for musical aesthetics, try modifying the `reward_music_theory` function!

## Helpful links

*   The code implements the model described in [this paper][our arxiv].
*   For more on DQN, see [this paper][dqn].
*   The DQN code was originally based on [this example][dqn ex].

[our arxiv]: https://arxiv.org/pdf/1611.02796v2.pdf
[blog post]: https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/
[ipynb]: https://nbviewer.jupyter.org/github/tensorflow/magenta/tree/master/magenta/models/rl_tuner/rl_tuner.ipynb
[note rnn ckpt]: http://download.magenta.tensorflow.org/models/rl_tuner_note_rnn.ckpt
[magenta pretrained]: https://github.com/magenta/magenta/tree/master/magenta/models/melody_rnn#pre-trained
[dqn ex]: https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/
[g learning]: https://arxiv.org/pdf/1512.08562.pdf
[psi learning]: http://homepages.inf.ed.ac.uk/svijayak/publications/rawlik-RSS2012.pdf
[dqn]: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
