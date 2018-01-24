## MusicVAE: A hierarchical recurrent variational autoencoder for music.

MusicVAE learns a latent space of musical sequences, providing different modes
of interactive musical creation, including:

* random sampling from the prior distribution,
* interpolation between existing sequences,
* manipulation of existing sequences via a [latent constraint model](https://goo.gl/STGMGx).

For short sequences (e.g., 2-bar "loops"), we use a bidirectional LSTM encoder
and LSTM decoder. For longer sequences, we use a novel hierarchical LSTM
decoder, which helps the model learn longer-term structures.

We also model the interdependencies among instruments by training multiple
decoders on the lowest-level embeddings of the hierarchical decoder.

Representations of melodies/bass-lines and drums are based on those used
by [MelodyRNN](/magenta/models/melody_rnn) and
[DrumsRNN](/magenta/models/drums_rnn).

For additional model details, see our [paper](https://nips2017creativity.github.io/doc/Hierarchical_Variational_Autoencoders_for_Music.pdf).

### How To Use

#### Colab Notebook w/ Pre-trained Models
The easiest way to get started using a MusicVAE model is via our
[Colab Notebook](https://colab.research.google.com/notebook#fileId=/v2/external/notebooks/magenta/music_vae/music_vae.ipynb).
The notebook contains instructions for sampling interpolating, and manipulating
musical sequences with pre-trained MusicVAEs for melodies, drums, and
three-piece "trios" (melody, bass, drums) of varying lengths.

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

Finally, you should also launch an evaluation job (using `--mode=eval` with a
heldout dataset) in order to compute metrics such as accuracy and to avoid
overfitting.

Once your model has trained sufficiently, you can load the checkpoint into the
[Colab Notebook](https://colab.research.google.com/notebook#fileId=/v2/external/notebooks/magenta/music_vae/music_vae.ipynb) to do inference and produce audio outputs.
