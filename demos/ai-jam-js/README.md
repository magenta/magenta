## A.I. Jam

An interactive A.I. jam session.

## ABOUT

This interface uses the frontend from the [A.I. Duet experiment](https://github.com/googlecreativelab/aiexperiments-ai-duet), combined with the [Magenta MIDI interface](/magenta/interfaces/midi/README.md) to recreate the experience of the award-winning [Magenta 2016 NIPS Demo](https://magenta.tensorflow.org/2016/12/16/nips-demo) in your browser.

This is not an official Google product.

## CREDITS

Original interface built by [Yotam Mann](https://github.com/tambien) with friends on the Magenta and Creative Lab teams at Google using [TensorFlow](https://tensorflow.org), [Tone.js](https://github.com/Tonejs/Tone.js) and open-source tools from the [Magenta](https://magenta.tensorflow.org/) project.

## GETTING STARTED

Install [Flask](http://flask.pocoo.org/) and [Magenta](/README.md#Installation) (v0.1.15 or greater). Also download the following pre-trained models, and save them to this directory.

* [Attention RNN](http://download.magenta.tensorflow.org/models/attention_rnn.mag)
* [Pianoroll RNN-NADE](http://download.magenta.tensorflow.org/models/pianoroll_rnn_nade.mag)
* [Drum Kit RNN](http://download.magenta.tensorflow.org/models/drum_kit_rnn.mag)

Then launch the experience from the command line:

```bash
sh RUN_DEMO.sh
```

## BUILDING

If you'd like to make modifications to this code, first make sure you have [Node.js](https://nodejs.org) 6 or above installed. And then install of the dependencies of the project and build the code by typing the following in the terminal:

```bash
cd static
npm install
node_modules/.bin/webpack -w
```

## MIDI SUPPORT

The A.I. Jam supports MIDI keyboard input using [Web Midi API](https://webaudio.github.io/web-midi-api/) and the [WebMIDI](https://github.com/cotejp/webmidi) library.

## PIANO KEYBOARD

The piano can also be controlled from your computer keyboard thanks to [Audiokeys](https://github.com/kylestetz/AudioKeys). The center row of the keyboard is the white keys.

## CONTROLS

Currently, the only way to change settings for the models is to use the keyboard
shortcuts below.

For now, it is recommended that you open the Developer JavaScript Console in
Chrome to view changes to settings based on your use of these shortcuts.

| Key        | Action |
|------------|--------|
| Z          | Toggles the metronome. |
| TAB        | Toggles between piano and drums. |
| LEFT/RIGHT | Cycles through available models. |
| UP/DOWN    | Adjusts sampling 'temperature'. Higher temperatures sound more random. |
| SPACE      | Toggles looping of AI sequence. |
| M          | Mutates AI sequence. |
| 0-9        | Sets AI response duration (in bars). 0 matches your input. |
| SHIFT + 0-9| Sets input sequence duration (in bars). 0 matches your input. |
| DELETE     | Stops current AI playback. |
| X          | Toggles "solo mode", which stops AI from listening to inputs. |

## AUDIO SAMPLES

Audio synthesized from [FluidR3_GM.sf2](http://www.musescore.org/download/fluid-soundfont.tar.gz) ([Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)).
