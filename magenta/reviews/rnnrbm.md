----
## Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription

The authors of [Boulanger-Lewandowski 2012](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf) describe a series of powerful sequential generative models, including the RNN-RBM. We can think of this powerful model as a series of RBMs whose parameters are determined by an RNN. Each RBM in the sequence is capable of modeling a complex and high dimensional probability distribution, and the RNN conditions each distribution on those of the previous time steps. 

### Architecture
The architecture of the RNN-RBM is not tremendously complicated. Each RNN hidden unit is paired with an RBM. RNN hidden unit h(t) takes input from data vector v(t) as well as from hidden unit h(t-1). The outputs of hidden unit h(t) are the parameters of RBM(t+1), which takes as input data vector v(t+1). 

![The parameters of each RBM are determined by the output of the RNN](assets/rnnrbm_color.png)

In [Boulanger-Lewandowski 2012](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf), all of the RBMs share the same weight matrix, and only the hidden and visible bias vectors are determined by the outputs of h(t). With this convention, the role of the RBM weight matrix is to specify a consistent prior on all of the RBM distributions, and the role of the bias vectors is to communicate temporal information. 

![The parameters of the model. This figure is from the paper](assets/rnnrbm_figure.png)

##### Generation
To generate a sequence with the RNN-RBM, we prime the RNN and repeat the following procedure:

- Use the RNN-to-RBM weight and bias matrices and the state of RNN hidden unit h(t-1) to determine the bias vectors for RBM(t) 
    ![The outputs of the RNN are the bias vectors of the RBM](assets/get_bias.png)
- Perform [Gibbs Sampling](http://stats.stackexchange.com/questions/10213/can-someone-explain-gibbs-sampling-in-very-simple-words) to sample from RBM(t) and generate v(t)
    ![Repeat this process k times, and then v(t) is the visible state at the end of the chain](assets/gibbs.png)
- Use v(t), the state of RNN hidden unit h(t-1) and the weight and bias matrices of the RNN to determine the state of RNN hidden unit h(t)
    ![Compute the hidden state at time t](assets/get_hidden.png)

##### Cost Function

The cost function for the RNN-RBM is the [contrastive divergence](http://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf) estimation of the negative log likelihood of the data vector v(t), as computed from RBM(t). In practice we compute this by:

-  Performing Gibbs sampling to sample output(t) (written as v(t)*) from RBM(t)
-  Taking the difference between the free energies of v(t) and v(t)*.

Then the gradient of the loss is ![We pass this loss back with BPTT](assets/grad_loss.png)

### Music

The authors of [Boulanger-Lewandowski 2012](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf) use the RNN-RBM to generate [polyphonic](https://en.wikipedia.org/wiki/Polyphony) music. In their representation, the inputs to the RNN and the outputs of the RBM are lines in a [midi](https://en.wikipedia.org/wiki/MIDI) piano roll. Some of the music that they've generated is below. You can find more files on [their website](http://www-etud.iro.umontreal.ca/~boulanni/icml2012)

<audio src="assets/4_RNN-RBM.mp3" controls preload></audio>
<audio src="assets/5_RNN-RBM.mp3" controls preload></audio>
<audio src="assets/6_RNN-RBM.mp3" controls preload></audio>

### Extensions
It's not too difficult to think of possible extensions to the RNN-RBM architecture. The authors utilize [Hessian-Free optimization](http://www.icml-2011.org/papers/532_icmlpaper.pdf) to help train the RNN to better recognize long term dependencies. Another tool that is very useful for representing long term dependencies is [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). It's reasonable to suspect that converting the RNN cells to LSTM cells would improve the model's ability to represent longer term patterns in the sequence, such as in [this paper](http://www.ijcai.org/Proceedings/15/Papers/582.pdf). 

Another way to increase the modelling power of the model is to replace the RBMs with a more complex model. For example, a [Deep Belief Network](https://www.cs.toronto.edu/~hinton/nipstutorial/nipstut3.pdf) is an unsupervised neural network architecture formed from multiple RBMs stacked on top of another. Unlike RBMs, DBNs can take advantage of their multiple layers to form hierarchical representations of data. In [this paper](http://www.academia.edu/16196335/Modeling_Temporal_Dependencies_in_Data_Using_a_DBN-LSTM) the authors combine DBNs and LSTMs to model high dimensional temporal sequences and generate music.
