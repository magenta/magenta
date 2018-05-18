## MusicVAE: A hierarchical recurrent variational autoencoder for music.

[MusicVAE](https://g.co/magenta/music-vae) learns a latent space of musical sequences, providing different modes
of interactive musical creation, including:

* random sampling from the prior distribution,
* interpolation between existing sequences,
* manipulation of existing sequences via attribute vectors or a [latent constraint model](https://goo.gl/STGMGx).

For short sequences (e.g., 2-bar "loops"), we use a bidirectional LSTM encoder
and LSTM decoder. For longer sequences, we use a novel hierarchical LSTM
decoder, which helps the model learn longer-term structures.

We also model the interdependencies among instruments by training multiple
decoders on the lowest-level embeddings of the hierarchical decoder.

Representations of melodies/bass-lines and drums are based on those used
by [MelodyRNN](/magenta/models/melody_rnn) and
[DrumsRNN](/magenta/models/drums_rnn).

For additional details, see our [blog post](https://g.co/magenta/music-vae) and [paper](https://goo.gl/magenta/musicvae-paper).

### How To Use

#### Colab Notebook w/ Pre-trained Models
The easiest way to get started using a MusicVAE model is via our
[Colab Notebook](https://g.co/magenta/musicvae-colab).
The notebook contains instructions for sampling interpolating, and manipulating
musical sequences with pre-trained MusicVAEs for melodies, drums, and
three-piece "trios" (melody, bass, drums) of varying lengths.

#### JavaScript w/ Pre-trained Models
We have also developed [MusicVAE.js](https://goo.gl/magenta/musicvae-js), a JavaScript API for interacting with
MusicVAE models in the browser. Existing applications built with this library include:

* [Beat Blender](https://g.co/beatblender) by [Google Creative Lab](https://github.com/googlecreativelab)
* [Melody Mixer](https://g.co/melodymixer) by [Google Creative Lab](https://github.com/googlecreativelab)
* [Latent Loops](https://goo.gl/magenta/latent-loops) by [Google Pie Shop](https://github.com/teampieshop)
* [Neural Drum Machine](https://codepen.io/teropa/pen/RMGxOQ) by [Tero Parviainen](https://github.com/teropa)

Learn more about the API in its [repo](https://goo.gl/magenta/musicvae-js).

#### Training Your Own MusicVAE

If you'd like to train a model on your own data, you will first need to set up
your [Magenta environment](/README.md). Next, convert a collection of MIDI files
into NoteSequences following the instructions in
[Building your Dataset](/magenta/scripts/README.md). You can then choose one of
the pre-defined Configurations in [configs.py](configs.py) or define your own.
Finally, you must execute the [training script](train.py). Below is an example
command, training the `cat-mel_2bar_small` configuration and assuming your
examples are stored at `/tmp/music_vae/mel_train_examples.tfrecord`.

```
music_vae_train \
--config=cat-mel_2bar_small \
--run_dir=/tmp/music_vae/ \
--mode=train \
--examples_path=/tmp/music_vae/mel_train_examples.tfrecord
```

You will likely need to adjust some of the hyperparamters with the `--hparams`
flag for your particular train set and hardware. For example, if the default
batch size of a config is too large for your GPU, you can reduce the batch size
and learning rate by setting the flag as follows:

```
--hparams=batch_size=32,learning_rate=0.0005
```

These models are particularly sensitive to the `free_bits` and `max_beta`
hparams. Decreasing the effect of the KL loss (by increasing `free_bits` or
decreasing `max_beta`) results in a model that produces better reconstructions,
but with potentially worse random samples. Increasing the effect of the KL loss
typically results in the opposite. The default config settings of these hparams
are an attempt to reach a balance between good sampling and reconstruction,
but the best settings are dataset-dependent and will likely need to be adjusted.

Finally, you should also launch an evaluation job (using `--mode=eval` with a
heldout dataset) in order to compute metrics such as accuracy and to avoid
overfitting.

Once your model has trained sufficiently, you can load the checkpoint into the
[Colab Notebook](https://goo.gl/magenta/musicvae-paper) to do inference and
produce audio outputs.
