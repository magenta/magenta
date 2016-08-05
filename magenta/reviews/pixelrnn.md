<p align="center">
  <img src="assets/pixelrnn_figure6.png">
</p>
Image source: [Pixel Recurrent Neural Networks, Figure 6. Van den Oord et. al.](https://arxiv.org/abs/1601.06759)

Pixel Recurrent Neural Networks
===============================
(Review by [Kyle Kastner](https://github.com/kastnerkyle))

The [Pixel Recurrent Neural Networks paper](https://arxiv.org/abs/1601.06759), by Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu, combines a number of techniques to generate the above images, introducing several new generative models (PixelCNN and two types of PixelRNN) in the process. These models greatly improve the state of the art in image generation using a combination of new innovations and smart integration of existing techniques, and are a master class in how to build generative models. It also won a [Best Paper award at ICML 2016](http://icml.cc/2016/?page_id=2009).

The specific images above are from a PixelRNN model trained on about 1 million *32 x 32* pixel color images derived from ImageNet. Generating images at the pixel level can be done in many different ways, but one common method (as seen in [NADE](http://www.dmi.usherb.ca/~larocheh/publications/aistats2011_nade.pdf) and [MADE](https://arxiv.org/abs/1502.03509)) is to choose a value for a pixel and use that to condition the next prediction. To accommodate this assumption, some ordering of the pixels is assumed, e.g. top left to bottom right and row-wise.

In probability terms, this approach uses the [chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)) to reinterpret the joint probability of an image<br>
**p(image) = p(x_0, x_1, x_2 … x_n)**, where **x_*** are the individual pixels in the image in a specific order,
as a product of conditional distributions, **p(image) = p(x_0) * p(x_1 | x_0) * p(x_2 | x_1, x_0)...** .

This process assigns an ordering among the pixels, allowing the image to be sampled and processed sequentially. Note that this ordering is one of many possible. Ensembling of several orderings was explored in great detail in both [NADE and MADE](http://videolectures.net/deeplearning2015_larochelle_deep_learning/), and related ideas about exploring structure using splitting and masking can also be seen in [NICE](https://arxiv.org/abs/1410.8516) and [Real NVP](https://arxiv.org/abs/1605.08803). Combined with efficient convolutional preprocessing, both PixelCNN and PixelRNN use this “product of conditionals” approach to great effect. This conditional dependency chain is straightforward to model with an RNN, but how do they get the same context with convolutional processing?

The authors solve this by masking the context each convolution can access. Masks are a clever way to model sequential conditional probability without any explicit recurrence, and were used to great effect in MADE. We will see that PixelCNN (a convolutional neural network) uses masks to capture a context window which increases over depth, but still has the correct conditional dependencies. PixelRNN (a recurrent neural network) also uses convolutional masking as input to the recurrent state for computational efficiency, rather than working directly on a pixel by pixel basis. The RNN model requires more processing time than the purely convolutional model per layer, but the combination of convolutional and [recurrent networks](http://www.deeplearningbook.org/contents/rnn.html) allows for fewer layers overall.

Conditional Probabilities by Masking
------------------------------------

There are two types of masks used in this work, **A** and **B**. A high level description for the construction of the masks can be seen in the following figures. The black and white images show a *5 x 5* filter where the number of input channels is three and the number of output channels is three. The left hand axis corresponds to a visualization of the mask from *R*, *G*, or *B* in the higher layer in *Figure 4* of the paper, back down to the data or activations in the layer below. Generalizing to more filter maps repeats this construction. For example, with *1024* feature maps there would be *113* sets of *9* masks (the 3 x 3 set of channel interactions), plus *7* more maps from the next “mask cycle”.

<p align="center">
  <img src="assets/pixelrnn_masks_highlevel.png">
</p>
Image source: [Pixel Recurrent Neural Networks, Figure 4. Van den Oord et. al.](https://arxiv.org/abs/1601.06759)

Mask A |  Mask B
:-----:|:-------:
<p align="left"><img src="assets/pixelrnn_masks_A.png", height="500", width="500"/></p> | <p align="right"><img src="assets/pixelrnn_masks_B.png", height="500", width="500"/></p>

Both masks are carefully constructed to ensure that the prediction for a pixel is never a function of its own input value. The key difference between mask **A** and mask **B** is whether the pixel being predicted is turned “on” in the center of the mask. Mask **A** is responsible for ensuring that the current pixel in question does not contribute to the prediction. With mask **A** eliminating the problem connection from the network input, the rest of the subsequent layer masks can then be type **B**, where self-connection on the current channel is allowed. This difference is extremely subtle, but key to how this model works.

Applying the masks to the filter weights before convolving will result in the correct dependency field. Some example code to generate these masks in numpy, by [Ishaan Gulrajani](https://github.com/igul222), can be found [here](https://github.com/igul222/pixel_rnn). That mask generation was the basis for the visualization code used to generate the above image.
Ishaan's code also implements PixelCNN and one form of PixelRNN on MNIST with reasonably good scores. Though it is not an exact replication of the paper, it is a useful reference for understanding these architectures in code. It also highlights how simple their implementation can be.

Models
------

For a basic PixelCNN, the overall architecture is straightforward - stack some number of masked convolutions with ReLU activations, being sure to use mask **A** as the input layer mask and mask **B** for each subsequent layer, then use *3 x 256* feature maps (in the RGB pixel case) for the last *1 x 1* convolutional layer and take a per pixel cross-entropy cost at the output.

The two types of PixelRNN models are Row LSTM and Diagonal BiLSTM. Both use masked convolutions to precompute the input gates for an LSTM. Diagonal BiLSTM computes these convolutions in a linearly mapped space, using a [skew mapping](https://en.wikipedia.org/wiki/Shear_mapping) so that the bidirectional network over these masked convolutions can capture the full, ideal context mapping at every layer. The result can then be “unskewed” using the inverse transform and repeated in subsequent layers.

Additionally all of these architectures can use modern techniques like residual connections, skip connections, gating over depth, and multi-scale architectures using learned convolutional upsampling. However the Diagonal BiLSTM without any of these tricks already sets the state of the art for MNIST, and a much larger model with most of these extra enhancements trounces the previous state of the art for CIFAR-10, though the authors don’t give details on the particular architecture used in the experiments listed in *Table 5*.

Understanding Context
---------------------

All three models described in this paper (PixelCNN, Row LSTM, and Diagonal BiLSTM) use masked convolutions to control the context and channels seen at each step of the model.
Row LSTM and Diagonal BiLSTM use additional recurrent processing to capture a larger immediate context window, rather than having context grow purely over depth.
In the case of the Diagonal BiLSTM in particular, each RNN position sees the entire available context at every layer. These “context windows” can be seen in *Figure 2* of the paper, and are also shown below.

On the far left is the ideal context, and on the right is the context seen by a Diagonal BiLSTM with convolutional kernel size *2*.
The middle is the context seen by a particular position in Row LSTM with convolutional kernel size *3*, with a small one pixel addition (in green) to the figure from the paper due to masked input convolution in the center context.
This neighbor pixel is important so that the generations at sample time remain vertically and horizontally consistent (from personal conversation, [Laurent Dinh](https://github.com/laurent-dinh)).
Without the green pixel there would be no context about what has previously been seen horizontally, which would make images with attributes like straight horizontal lines more difficult to model.
The Diagonal BiLSTM context exactly matches the ideal context, which is one reason why even a *1* layer Diagonal BiLSTM has great results.

<p align="center">
  <img src="assets/pixelrnn_full_context.png">
</p>

Image modified from: [Pixel Recurrent Neural Networks, Figure 2. Van den Oord et. al.](https://arxiv.org/abs/1601.06759)

Another feature of this model is exact likelihoods (versus the lower bound given by [DRAW](https://github.com/tensorflow/magenta/blob/master/magenta/reviews/draw.md)) due to the direct use of a categorical cross-entropy cost per pixel. This is the second key piece of this work, as using a categorical cross-entropy cost and a softmax allows the model to directly place probability mass only on valid pixel values in the range *[0, 255]*. In addition, this type of cost is common in classification tasks and is well explored with existing optimization and regularization methods.

Image Generation
----------------
Sampling PixelCNN works by running the model on a blank image and first predicting the top left pixel’s red channel. Run the model again with this new input and predict the green channel for the first pixel. Repeat for the blue channel, then continue the same process for the second pixel and so on. Sampling Pixel RNN is performed similarly, though the specific architecture used to generate the predictions is different. In both cases, sampling is a sequential process.

The movement toward iterative generation (seen in many of the [other reviews](https://github.com/tensorflow/magenta/blob/master/magenta/reviews/README.md)) has resulted in several powerful generative models. Breaking the generative process into steps can allow for more expressive models, which take into account the generations at previous steps for high quality outputs.

Footnote
--------

Directly modeling all of the conditional probabilities for *32 x 32* pixel color images would result in a linear chain of 3072 conditional probabilities.
This direct computation is how NADE worked. PixelCNN instead exploits topological structure to grow the dependency chain over depth. This leads to an effective sqrt(n) dependency from local context to global (from personal discussion, [Laurent Dinh](https://github.com/laurent-dinh)).

This naturally encodes what we see and know about images. Real world images feature strong local correlations and correspondingly have close connection over depth in a PixelCNN (within one or two layers depending on convolutional kernel size), while looser global structure contexts are found [deeper in the network](https://kaggle2.blob.core.windows.net/forum-message-attachments/69182/2287/A%20practical%20theory%20for%20designing%20very%20deep%20convolutional%20neural%20networks.pdf?sv=2012-02-12&se=2016-07-01T04%3A02%3A12Z&sr=b&sp=r&sig=aP8jLZEO6YmPwYps1NiVDwyLlxy5tCJjCsP%2B2FIwkU0%3D).
This growth from local to global dependencies may improve the model from an optimization and learning perspective, and the authors explore the modeling capacity of the PixelCNN in detail in a [more recent paper](https://arxiv.org/abs/1606.05328).

Acknowledgements
----------------

Special thanks to Laurent Dinh, Tim Cooijmans, Jose Sotelo, Natasha Jaques, Cinjon Resnick, Anna Huang, hardmaru, Aaron Courville, the MILA speech synthesis team, and the Magenta team for helpful discussions on various aspects of this paper. Also thanks to the authors of the paper (Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu) for publishing such interesting work.

Related Work and References
---------------------------
[Probability chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)), basic probability manipulations.

[Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759), the main paper for this post.

[Conditional Pixel CNN](https://arxiv.org/abs/1606.05328), a recent followup by Van den Oord et. al.

[Generative Image Modeling Using Spatial LSTMs](http://arxiv.org/abs/1506.03478), a precursor to PixelRNN for texture generation, by Theis and Bethge.

[Multi-Dimensional Recurrent Neural Networks](https://arxiv.org/abs/0705.2011), baseline approach to using RNNs for multidimensional data, by Graves, Fernandez, and Schmidhuber.

[Grid LSTM](https://arxiv.org/abs/1507.01526), a recent followup for stronger multidimensional RNNs, by Kalchbrenner, Danihelka, and Graves.

[ReNet](https://arxiv.org/abs/1505.00393), using "decomposed" multidimensional RNNs for image prediction, Visin et. al.

[ReSeg](https://arxiv.org/abs/1505.00393), semantic segmentation using multidimensional RNNs, Visin et. al.

[H-ReNet](http://arxiv.org/abs/1603.04871), a high-performance approach for segmentation using multidimensional RNN, Yan et. al.

[Deep Learning Book](http://www.deeplearningbook.org/contents/rnn.html), indepth discussion of RNNs from Goodfellow, Courville, Bengio.

[Open source code in Theano by Ishaan Gulrajani](https://github.com/igul222/pixel_rnn), a great reference implementation of PixelCNN and DiagonalBiLSTM.

[Skew mapping](https://en.wikipedia.org/wiki/Shear_mapping), used for the full context in Diagonal BiLSTM.

[Residual connections](https://arxiv.org/abs/1512.03385), a key part of modern convolutional networks, by He et. al.

[Introduction to using skip connections](http://arxiv.org/abs/1308.0850), a key part of deep RNNs, by Graves.

[Highway Networks](https://arxiv.org/abs/1505.00387), the precursor to residual connections for extreme depth, by Srivistava et. al..

[Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](http://arxiv.org/abs/1406.2283), an example of conditional predictions using convolutional networks, by Eigen, Puhrsch, and Fergus.

[A Simple Way To Initialize Recurrent Networks of Rectified Linear Units](https://arxiv.org/abs/1504.00941), per pixel MNIST modeling with RNNs, by Le et. al..

[A practical guide to sizing very deep convolutional networks](https://kaggle2.blob.core.windows.net/forum-message-attachments/69182/2287/A%20practical%20theory%20for%20designing%20very%20deep%20convolutional%20neural%20networks.pdf?sv=2012-02-12&se=2016-07-01T04%3A02%3A12Z&sr=b&sp=r&sig=aP8jLZEO6YmPwYps1NiVDwyLlxy5tCJjCsP%2B2FIwkU0%3D),
a guide to receptive fields and depth in CNNs.

[Forum post on convolutional network sizing](https://www.kaggle.com/c/datasciencebowl/forums/t/13166/happy-lantern-festival-report-and-code/69196), forum post referencing the above PDF.

[NADE](http://www.dmi.usherb.ca/~larocheh/publications/aistats2011_nade.pdf),
using autoregressive ordering for exact likelihoods in a generative setting, by Larochelle and Murray.

[MADE](https://arxiv.org/abs/1502.03509),
using masks to imply an ensemble of autoregressive orderings at once, by Germain et. al.

[NADE, MADE Slides](http://videolectures.net/deeplearning2015_larochelle_deep_learning/),
presentation by Hugo Larochelle at the Deep Learning Summer School in 2015.

[NICE](https://arxiv.org/abs/1410.8516), generative modeling with exact likelihoods, by Dinh et. al.

[Real NVP](https://arxiv.org/abs/1605.08803), the followup to NICE, by Dinh et. al.

[Strided convolution guide](https://arxiv.org/abs/1603.07285), guide to the math and
concepts behind strided convolutions for generating larger targets with a CNN.
