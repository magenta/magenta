### Recurrent Reinforcement Learning for Generating Melodies

This code implements the models described in [this research paper][our arxiv].
The idea is to take an LSTM that has been trained to predict the next note in a
monophonic melody --- called a Note RNN --- and enhance it using reinforcement 
learning (RL). 

The RLTuner class implements a Deep Q Network (DQN), in which the Q network 
learns the reward value of taking actions (playing notes) given the state of the 
environment (the melody composed so far). The reward that the network learns 
comes from two sources: 1) a set of music theory reward functions, and 2) the 
output of a trained Note RNN, which gives p(a|s), the probability of playing the 
next note *a* given the state of the composition *s*, as originally learned from 
data. This combination allows the model to maintain what it learned from data, 
while constraining it to conform to a set of music theory rules. 

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
improved and extended.

In addition to the normal Q function, this code provides the ability to train 
the network with the [Psi learning][psi learning] and [G learning][g learning]
functions, which can be set with the **algorithm** hyperparameter. For details 
on each algorithm, see [our paper][our arxiv].

## Understanding the code
*   To initialize the RLTuner, pass it a directory containing a checkpoint of a 
	trained Note RNN using the `note_rnn_checkpoint_dir` parameter. It will 
	load the **q_network**, **target_q_network**, and **reward_rnn**, from this 
	checkpoint.

*	The tensorflow graph structure is defined in the **build_graph** function.

*	Use the **generate_music_sequence** function to generate a melody using the 
	original Note RNN. If you set the **visualize_probs** parameter to True, it 
	will plot the note probabilities of the model over time.

*	Train the model using the **train** function. It will continuously place 
	notes using **action**, receive rewards using **collect_reward**, and save 
	these experiences using **store**.

*	The network weights are updated using **training_step** which samples 
	minibatches of experience from the model's **experience** buffer and uses 
	this to compute gradients based on the loss function in **build_graph**.

*	During training, the function **evaluate_model** is occasionally run to 
	test how much reward the model receives from both the Reward RNN and the 
	music theory functions.

*	After the model is trained, you can use the **save_model_and_figs** function
	to save a checkpoint of the model and a set of figures of the rewards over 
	time. 

*	Finally, use **generate_music_sequence** again to see how the model's songs
	have improved with training!

## How to run the model
Or, you can simply run:

```
bazel run magenta/models/rl_tuner:rl_tuner_train -- 
--note_rnn_checkpoint_dir 'path' --midi_primer 'primer.mid' 
```
## Helpful links

*   The code implements the model described in [this paper][our arxiv]
*   The DQN code was originally based on [this example][dqn ex]

[our arxiv]: https://arxiv.org/abs/comingsoon
[dqn ex]: https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/
[g learning]: https://arxiv.org/pdf/1512.08562.pdf
[psi learning]: http://homepages.inf.ed.ac.uk/svijayak/publications/rawlik-RSS2012.pdf