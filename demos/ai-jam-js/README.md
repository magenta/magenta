## A.I. Jam

An interactive A.I. jam session.

## ABOUT

This interface uses the frontend from the [A.I. Duet experiment](https://github.com/googlecreativelab/aiexperiments-ai-duet), combined with the [Magenta MIDI interface](/magenta/interfaces/midi/README.md) to recreate the experience of the award-winning [Magenta 2016 NIPS Demo](https://magenta.tensorflow.org/2016/12/16/nips-demo) in your browser.

## CREDITS

Original interface built by [Yotam Mann](https://github.com/tambien) with friends on the Magenta and Creative Lab teams at Google using [TensorFlow](https://tensorflow.org), [Tone.js](https://github.com/Tonejs/Tone.js) and open-source tools from the [Magenta](https://magenta.tensorflow.org/) project.

## GETTING STARTED

### Pre-built Version

Install [Flask](http://flask.pocoo.org/) and [Magenta](/README.md#Installation) (v0.1.15 or greater). Also download the following pre-trained models, and save them to this directory.

* [Attention RNN](http://download.magenta.tensorflow.org/models/attention_rnn.mag)
* [Pianoroll RNN-NADE](http://download.magenta.tensorflow.org/models/pianoroll_rnn_nade.mag)
* [Drum Kit RNN](http://download.magenta.tensorflow.org/models/drum_kit_rnn.mag)

Then launch the interface from the command line:

```bash
sh RUN_DEMO.sh
```

### Development

If you'd like to make modifications to this code, first make sure you have [Node.js](https://nodejs.org) 6 or above installed. You can then install of the dependencies of the project and build the code by typing the following in the terminal:

```bash
cd static
npm install
node_modules/.bin/webpack
```

Adding a `-w` flag to the final command will cause webpack to continuously re-compile the JavaScript code as you make changes.

## MIDI SUPPORT

The A.I. Jam supports MIDI keyboard input using [Web Midi API](https://webaudio.github.io/web-midi-api/) and the [WebMIDI](https://github.com/cotejp/webmidi) library.

MIDI input is routed from the browser to MIDI ports "magenta_piano_in" and "magenta_drums_in", on which two instances of the `magenta_midi` binary are listening; one for piano and the other for drums. Both of these instances output responses to the "magenta_out" port, which the browser is listening to for playback.

The two `magenta_midi` instances are kept in sync by a running `midi_clock` binary that outputs a metronome using MIDI control change messages on the "magenta_clock" port. You can listen to the metronome by pressing the `Z` key while the browser is in focus.

If you'd like to use your own device or software for synthesis, you can simply mute the browser tab and route these MIDI ports appropriately to your device.

## PIANO KEYBOARD

The piano can also be controlled from your computer keyboard thanks to [Audiokeys](https://github.com/kylestetz/AudioKeys). The center row of the keyboard is the white keys.

## CONTROLS

Currently, the only way to change settings for the models is to use the keyboard
shortcuts below.

| Key              | Action |
|------------------|--------|
| `Z`              | Toggles the metronome. |
| `Q`              | Toggles between piano and drums. |
| `LEFT`/`RIGHT`   | Cycles through available models. |
| `UP`/`DOWN`      | Adjusts sampling 'temperature'. Higher temperatures sound more random. |
| `SPACE`          | Toggles looping of AI sequence. |
| `M`              | Mutates AI sequence. |
| `0`-`9`          | Sets AI response duration (in bars). 0 matches your input. |
| `SHIFT` + `0`-`9`| Sets input sequence duration (in bars). 0 matches your input. |
| `DELETE`         | Stops current AI playback. |
| `X`              | Toggles "solo mode", which stops AI from listening to inputs. |

## AUDIO SAMPLES

Audio synthesized from [FluidR3_GM.sf2](http://www.musescore.org/download/fluid-soundfont.tar.gz) ([Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)).
