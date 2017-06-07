## A.I. Duet

A piano that responds to you.

## About

This experiment lets you make music through machine learning. A neural network was trained on many MIDI examples and it learned about musical concepts, building a map of notes and timings. You just play a few notes, and see how the neural net responds. 

[https://aiexperiments.withgoogle.com/ai-duet](https://aiexperiments.withgoogle.com/ai-duet)

This is not an official Google product.

## CREDITS

Built by [Yotam Mann](https://github.com/tambien) with friends on the Magenta and Creative Lab teams at Google. It uses [TensorFlow](https://tensorflow.org), [Tone.js](https://github.com/Tonejs/Tone.js) and open-source tools from the [Magenta](https://magenta.tensorflow.org/) project. Check out more at [A.I. Experiments](https://aiexperiments.withgoogle.com).

## OVERVIEW

A.I. Duet is composed of two parts, the front-end which is in the `static` folder and the back-end which is in the `server` folder. The front-end client creates short MIDI files using the players's input which is sent to a [Flask](http://flask.pocoo.org/) server. The server takes that MIDI input and "continues" it using [Magenta](https://github.com/tensorflow/magenta) and [TensorFlow](https://www.tensorflow.org/) which is then returned back to the client. 

## INSTALLATION

Install [Flask](http://flask.pocoo.org/), Magenta and TensorFlow. Also download the [attention_rnn](http://download.magenta.tensorflow.org/models/attention_rnn.mag) bundle and put it in the `server` folder. 

Then run the server from the command line: 

```bash
cd server
python server.py
```

Then to build and install the front-end Javascript code, first make sure you have [Node.js](https://nodejs.org) 6 or above and [webpack](https://webpack.github.io/) installed. And then install of the dependencies of the project and build the code by typing the following in the terminal: 

```bash
cd static
npm install
webpack -p
```

You can now play with A.I. Duet at [localhost:8080](http://localhost:8080).

## MIDI SUPPORT

The A.I. Duet supports MIDI keyboard input using [Web Midi API](https://webaudio.github.io/web-midi-api/) and the [WebMIDI](https://github.com/cotejp/webmidi) library. 

## PIANO KEYBOARD

The piano can also be controlled from your computer keyboard thanks to [Audiokeys](https://github.com/kylestetz/AudioKeys). The center row of the keyboard is the white keys.

## AUDIO SAMPLES

Multisampled piano from [Salamander Grand Piano V3](https://archive.org/details/SalamanderGrandPianoV3) by Alexander Holm ([Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)).

String sounds from [MIDI.js Soundfonts](https://github.com/gleitz/midi-js-soundfonts) generated from [FluidR3_GM.sf2](http://www.musescore.org/download/fluid-soundfont.tar.gz) ([Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)).