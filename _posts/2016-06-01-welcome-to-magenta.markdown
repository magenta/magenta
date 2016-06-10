---
layout: post
title:  Welcome to Magenta!
date:   2016-06-01 08:00:00 -0700
permalink: welcome-to-magenta
author: douglaseck
tags: magenta
---

We’re happy to announce Magenta, a project from the [Google Brain
team](https://research.google.com/teams/brain/) that asks: Can we use
machine learning to create compelling art and music? If so, how? If
not, why not?  We’ll use [TensorFlow](https://www.tensorflow.org), and
we’ll release our models and tools in open source on our GitHub. We’ll
also post demos, tutorial blog postings, and technical papers. Soon
we’ll begin accepting code contributions from the community at
large. If you’d like to keep up on Magenta as it grows, you can follow
us on our [GitHub](https://github.com/tensorflow/magenta) and join our
[discussion
group](https://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

# What is Magenta?

Magenta has two goals. First, it’s a research project to advance the
state of the art in machine intelligence for music and art
generation. Machine learning has already been used extensively to
understand content, as in speech recognition or translation. With
Magenta, we want to explore the other side—developing algorithms that
can learn how to generate art and music, potentially creating
compelling and artistic content on their own.

Second, Magenta is an attempt to build a community of artists, coders,
and machine learning researchers. The core Magenta team will build
open-source infrastructure around TensorFlow for making art and music.
We’ll start with audio and video support, tools for working with
formats like MIDI, and platforms that help artists connect to machine
learning models.  For example, we want to make it super simple to play
music along with a [Magenta](https://www.youtube.com/watch?v=01V9S8_7A0I&feature=youtu.be) performance model.

We don't know what artists and musicians will do with these new tools,
but we're excited to find out. Look at the history of creative
tools. Daguerre and later Eastman didn’t imagine what [Annie
Liebovitz](https://en.wikipedia.org/wiki/Annie_Leibovitz) or [Richard
Avedon](https://en.wikipedia.org/wiki/Richard_Avedon) would accomplish
in photography. Surely Rickenbacker and Gibson didn’t have [Jimi
Hendrix](https://en.wikipedia.org/wiki/Jimi_Hendrix) or
[St. Vincent](https://en.wikipedia.org/wiki/St._Vincent_(musician)) in
mind.  We believe that the models that have worked so well in speech
recognition, translation and image annotation will seed an exciting
new crop of tools for art and music creation.

To start, Magenta is being developed by a small team of researchers
from the Google Brain team.  If you’re a researcher or a coder, you
can check out our alpha-version
[code](https://www.github.com/tensorflow/magenta). Once we have a
stable set of tools and models, we’ll invite external contributors to
check in code to our GitHub. If you’re a musician or an artist (or
aspire to be one—it’s easier than you might think!), we hope you’ll
try using these tools to make some noise or images or videos... or
whatever you like.

Our goal is to build a community where the right people are there to
help out.  If the Magenta tools don’t work for you, let us know.  We
encourage you to join our discussion list and shape how Magenta
evolves.  We'd love to know what you think of our work—as an artist,
musician, researcher, coder, or just an aficionado. You can follow our
progress and check out some of the music and art Magenta helps create
right here on this blog.  As we begin accepting code from community
contributors, the blog will also be open to posts from these
contributors, not just Google Brain team members.


# Research Themes 

We'll talk about our research goals in more depth later, via a series
of tutorial blog postings. But here’s a short outline to give an idea
of where we're heading.

## Generation 

Our main goal is to design algorithms that learn how to generate art
and music.  There's been a lot of great work in image generation from
neural networks, such as
[DeepDream](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html)
from A. Mordvintsev et al. at Google and [Neural Style
Transfer](http://arxiv.org/abs/1508.06576) from L. Gatys et al. at
U. Tübingen. We believe this area is in its infancy and expect to see
fast progress here. For those following machine learning closely, it
should be clear that this progress is already well underway.  But
there remain a number of interesting questions: How can we make models
like these truly
[generative](https://en.wikipedia.org/wiki/Generative_model)? How can
we better take advantage of user feedback?

## Attention and Surprise

It's not enough just to sample images or sequences from some learned
distribution.  Art is dynamic! Artists and musicians draw our
attention to one thing at the expense of another. They change their
story over time—is any Beatles album exactly like another?—and there's
always some element of surprise at play. How do we capture effects
like attention and surprise in a machine learning model? While we
don't have a complete answer for this question, we can point to some
interesting models such as the [Show, Attend and Tell
model](http://arxiv.org/abs/1502.03044) by Xu et al. from the [MILA
lab](https://mila.umontreal.ca/en/) in Montreal that learns to control
an attentional lens, using it to generate descriptive sentences of
images.

## Storytelling

This leads to perhaps our biggest challenge: combining generation,
attention, and surprise to tell a compelling story.  So much
machine-generated music and art is good in small chunks but lacks any
sort of long-term narrative arc. (To be fair, my own 2002 [music
generation
work](http://www.iro.umontreal.ca/~eckdoug/blues/index.html) falls
into this category).  Alternately, some machine generated content does
have long-term structure, but that structure is provided TO rather
than learned BY the algorithm. This is the case, for example, in David
Cope’s very interesting [Experiments in Musical Intelligence
(EMI)](http://artsites.ucsc.edu/faculty/cope/experiments.htm), in
which an AI model deconstructs compositions by human composers, finds
common signatures in them and recombines them into new works.  The
design of models that learn to construct long narrative arcs is
important not only for music and art generation but also areas like
language modeling, where it remains a challenge to carry meaning even
across a long paragraph, much less whole stories. Attention models
like the Show, Attend and Tell point to one promising direction, but
this remains a very challenging task.

## Evaluation

Evaluating the output of generative models is deceivingly
difficult. The time will come when Magenta has 20 different music
generation models available in open source.  How do we decide which
ones are good?  One option is to compare model output to training data
by measuring
[likelihood](https://en.wikipedia.org/wiki/Likelihood_function).  For
music and art, this doesn't work very well. As argued very nicely in
[A note on generative models](http://arxiv.org/abs/1511.01844) (Theis
et al.), it's easy to generate outputs that are close in terms of
likelihood, but far in terms of appeal (and vice versa). This
motivates work in artificial adversaries such as [Generative
Adversarial
Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
by Goodfellow et al. from MILA in Montreal. In the end, to answer the
evaluation question we need to get Magenta tools in the hands of
artists and musicians, and Magenta media in front of viewers and
listeners. As Magenta evolves, we'll be working on good ways to
achieve this.

## Other Google efforts

Finally, we want to mention other Google efforts and resources related
to Magenta.  The [Artists and Machine Intelligence
(AMI)](https://ami.withgoogle.com/) project is connecting with artists
to ask: What do art and technology have to do with each other? What is
machine intelligence, and what does ‘machine intelligence art’ look,
sound and feel like? Check out their
[blog](https://medium.com/artists-and-machine-intelligence) for more
about AMI.

The [Google Cultural
Institute](https://www.google.com/culturalinstitute/home) is fostering
the discovery of exhibits and collections from museums and archives
all around the world. Via their [Lab at the Cultural Institute](https://www.google.com/culturalinstitute/thelab/), they’re also
connecting directly with artists. As we make TensorFlow/Magenta the
best machine learning platform in the world for art and music
generation, we’ll work closely with both AMI and the Google Cultural
Institute to connect artists with technology. To learn more about our
various efforts, be sure to check out the [Google Research
Blog](http://googleresearch.blogspot.com/).

