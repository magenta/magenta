# Models

This directory contains Magenta models.

* [**Arbitrary Image Stylization**](/magenta/models/arbitrary_image_stylization): A machine learning system for performing fast artistic style transfer that may work on arbitrary painting styles.
* [**Coconet**](/magenta/models/coconet): Counterpoint by Convolution train a convolutional neural network to complete partial musical scores.
* [**Drums RNN**](/magenta/models/drums_rnn): Applies language modeling to drum track generation using an LSTM.
* [**GANSynth**](/magenta/models/gansynth): GANSynth is an algorithm for synthesizing audio with generative adversarial networks.
* [**Image Stylization**](/magenta/models/image_stylization): A "Multistyle Pastiche Generator" that generates artistics representations of photographs. Described in [*A Learned Representation For Artistic Style*](https://arxiv.org/abs/1610.07629).
* [**Improv RNN**](/magenta/models/improv_rnn): Generates melodies a la [Melody RNN](/magenta/models/melody_rnn), but conditions the melodies on an underlying chord progression.
* [**Melody RNN**](/magenta/models/melody_rnn): Applies language modeling to melody generation using an LSTM.
* [**Music VAE**](/magenta/models/music_vae): A hierarchical recurrent variational autoencoder for music.
* [**NSynth**](/magenta/models/nsynth): "Neural Audio Synthesis" as described in [*NSynth: Neural Audio Synthesis with WaveNet Autoencoders*](https://arxiv.org/abs/1704.01279).
* [**Onsets and Frames**](/magenta/models/onsets_frames_transcription): Automatic piano music transcription model as described in [*Onsets and Frames: Dual-Objective Piano Transcription*](https://arxiv.org/abs/1710.11153)
* [**Performance RNN**](/magenta/models/performance_rnn): Applies language modeling to polyphonic music using a combination of note on/off, timeshift, and velocity change events.
* [**Piano Genie**](/magenta/models/piano_genie): Piano Genie is a system for learning a low-dimensional discrete representation of piano music. It uses an encoder RNN to compress piano sequences (88 keys) into many fewer buttons (e.g. 8). A decoder RNN is responsible for converting the simpler sequences back to piano space.
* [**Pianoroll RNN-NADE**](/magenta/models/pianoroll_rnn_nade): Applies language modeling to polyphonic music generation using an LSTM combined with a NADE, an architecture called an RNN-NADE. Based on the architecture described in [*Modeling Temporal Dependencies in High-Dimensional Sequences:
Application to Polyphonic Music Generation and Transcription*](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf).
* [**Polyphony RNN**](/magenta/models/polyphony_rnn): Applies language modeling to polyphonic music generation using an LSTM. Based on the [BachBot](http://bachbot.com/) architecture described in [*Automatic Stylistic Composition of Bach Choralies with Deep LSTM*](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/156_Paper.pdf).
* [**RL Tuner**](/magenta/models/rl_tuner): Takes an LSTM that has been trained to predict the next note in a monophonic melody and enhances it using reinforcement learning (RL). Described in [*Tuning Recurrent Neural Networks with Reinforcement Learning*](https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/) and [*Sequence Tutor: Conservative Fine-Tuning of Sequence Generation Models with KL-control*](https://arxiv.org/abs/1611.02796)
* [**Score2Perf and Music Transformer**](/magenta/models/score2perf): Score2Perf is a collection of Tensor2Tensor problems for generating musical performances, either unconditioned or conditioned on a musical score.
* [**Sketch RNN**](/magenta/models/sketch_rnn): A recurrent neural network model for generating sketches. Described in [*Teaching Machines to Draw*](https://research.googleblog.com/2017/04/teaching-machines-to-draw.html) and [*A Neural Representation of Sketch Drawings*](https://arxiv.org/abs/1704.03477).
* [**SVG VAE**](/magenta/models/svg_vae): SVG VAE is a Tensor2Tensor problem for generating font SVGs. Described in [*A Learned Representation for Scalable Vector Graphics*](https://arxiv.org/abs/1904.02632).
