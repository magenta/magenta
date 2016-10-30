## Recurrent Reinforcement Learning for Generating Melodies

The MelodyQNetwork model is designed to apply reinforcement learning to a pre-
trained LSTM RNN that can predict the next note in a sequence. It applies
reinforcement learning via the classic Deep Q Learning method, in which a neural
network (in this case a recurrent LSTM), called the Q network, learns to predict
the value of taking each action given an input state.

The model is built on top of the MelodyRNN class defined in melody_rnn.py. A
MelodyRNN is a basic LSTM model designed to predict the next note given a
sequence of previous note. MelodyQ is composed of a 'q_network' MelodyRNN, a
'target_q_network' MelodyRNN, and a 'reward_rnn' MelodyRNN, all loaded from the
same checkpoint file.

In this model, the 'q_network' is used to estimate the value in terms of
expected reward of playing each note given the history of previous notes. During
training, the 'q_network' is used to choose the model's actions, and a reward
function is used to assign a reward for those actions given ideas from music
theory, the previous note, the key, the beat, etc. Several reward functions have
been written, and these could easily be improved and extended.

The music theory reward functions are designed to constrain the actions of the
network so that it chooses notes in accordance with a musical structure; for
example, notes within the same key. No reward or zero reward is given for notes
that are outside of what is defined acceptable according to the music theory
reward function. However, if the network chooses a note that is acceptable, what
reward should it receive? This is the function of the 'reward_rnn'. Because we
would still like the network's compositions to be influenced by what it has
learned from actual data, the reward it receives for playing acceptable notes
will be the output of the 'reward_rnn' for that note. The 'reward_rnn' remains
fixed; its weights are not updated. Therefore it allows us to train the model to
play compositions similar to what it has learned from data, so long as those
compositions conform to a musical structure defined by the reward function.

During training, the total reward for a given action is computed as the reward
received for that action, plus the maximum expected reward that can be obtained
from the next state that results from the current action. The gradient of the
difference between the 'q_network's estimated value for the action and the total
reward is used to train the model. The 'target_q_network' is used to estimate
the expected reward from the next state, to ensure stability. The
'target_q_network' is not updated from the gradients, but rather it is updated
gradually over time, based on the 'target_network_update_rate'.

The training loop continuously primes the q_network model for a new composition,
allows it to select notes, and computes and stores the associated reward and
LSTM state in an experience buffer. Random batches of (observation, state,
action, reward, new_observation, new_state) tuples are sampled from the
experience buffer to train the 'q_network' model. Initially, the model will
explore by acting randomly, gradually annealing the probability of taking a
random action over the course of the exploration period. After each composition
is finished the q_network is reset and primed again.

## Helpful links

*   The code implements the model described in [this paper][our arxiv]
*   The DQN code was originally based on [this example]
    (https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/)

[our arxiv]: https://arxiv.org/abs/comingsoon