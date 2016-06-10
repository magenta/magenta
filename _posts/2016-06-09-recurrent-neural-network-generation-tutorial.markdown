---
layout: post
title:  A Recurrent Neural Network Music Generation Tutorial
date:   2016-06-09 18:00:00 -0700
author: danabo
tags: model,melody,rnn,lstm
---

We are excited to release our first
[tutorial model](https://github.com/tensorflow/magenta/tree/master/magenta/models/basic_rnn),
a recurrent neural network that generates music. It serves as an end-to-end primer on how to build
a recurrent network in [TensorFlow](https://www.tensorflow.org). It also
demonstrates a sampling of what's to come in Magenta. In addition, we are
releasing code that converts MIDI files to a format that TensorFlow can
understand, making it easy to create training datasets from any collection of
MIDI files.

This tutorial will allow you to to generate music with a recurrent neural
network. It's purposefully a simple model, so don't expect stellar music
results. We'll post more complex models soon.


# Background on Recurrent Neural Networks

A recurrent neural network (RNN) has looped, or recurrent, connections which
allow the network to hold information across inputs. These connections can be
thought of as similar to memory. RNNs are particularly useful for learning
sequential data like music.

In TensorFlow, the recurrent connections in a graph are unrolled into an
equivalent feed-forward network. That network is then trained using a gradient
descent technique called backpropagation through time
([BPTT](https://en.wikipedia.org/wiki/Backpropagation_through_time)).

{% include image.html
url="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png"
description="An RNN's recurrent connection unrolled through time. Image courtesy
of Chris Olah." alt="Unrolled RNN" %}

There are endless ways that an RNN can connect back to itself with recurrent
connections. People typically stick to a few common patterns, the most common
being Long Short-Term Memory (LSTM) cells and Gated Recurrent Units (GRU). These
both have multiplicative gates that protect their internal memory from being
overwritten too easily, allowing them to handle longer sequences. We use LSTMs
in this model. To learn more about RNNs and specifically LSTMs, check out
[Chris Olah's fantastic post](http://colah.github.io/posts/2015-08-Understanding-LSTMs). Experts
in the field might also like to look at Goodfellow, Bengio and Courville's
[RNN chapter](http://www.deeplearningbook.org/contents/rnn.html) from their book
"Deep Learning."

# This Release

This RNN is the first in a series of models we will be releasing which predict
the next note given a sequence of previous notes. They do this by learning a
probability distribution over the next notes given all the previous notes. By
sampling from that distribution and feeding the chosen note back into the model
at the next step, the RNN can dream up an entire melody. Generative models are
typically unsupervised, meaning that there are samples but no labels. However we
turn the problem of melody generation into a supervised one by trying to predict
the next note in a sequence, that way labels can be derived from any dataset of
just music and nothing else. This allows us to use RNNs which are supervised
models.

It takes a bit of work to put together a training set of melodies, so we are
providing code that reads an archive of MIDI files and outputs monophonic melody
lines extracted from them in a format TensorFlow can understand. After you have
that ready, instructions to build and run the model are
[here](https://github.com/tensorflow/magenta/tree/master/magenta/models/basic_rnn).


# Feedback

As always, we are excited to hear from you. Let us know what you liked, didn't
like, and want to see in the future from Magenta. You can add some code to our
[GitHub](https://github.com/tensorflow/magenta) or join our
[discussion group](https://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).