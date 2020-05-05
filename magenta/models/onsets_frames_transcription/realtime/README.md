## Onsets and Frames: Realtime TFLite Demo

This is an experimental demo of an Onsets and Frames model running under
[TensorFlow Lite](https://www.tensorflow.org/lite). You can run it as a fun
demo or use it as a starting point for embedded applications.

This demo has some dependencies that the regular Magneta package does not.
To include them, use `pip install magenta[onsets_frames_realtime]`.

### Model

First, download one of the pre-trained models:

- [onsets_frames_wavinput.tflite](https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite) - Full model (most accurate).
- [onsets_frames_wavinput_uni.tflite](https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput_uni.tflite) - Unidirectional LSTM.
- [onsets_frames_wavinput_no_offset_uni.tflite](https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput_no_offset_uni.tflite) - Unidirectional LSTM, no offset stack (most efficient).

We're working on cleaning up and open sourcing the code to translate a regular
Onsets and Frames model to TFLite as was done to produce this model.

### Installation for regular Linux machines

(For installation on Raspberry Pi4, see below)
First, set up your [Magenta environment](/README.md).

Then, run the demo:

```
onsets_frames_transcription_realtime \
  --model_path /path/to/onsets_frames_wavinput.tflite
```

### Installation on Raspberry Pi4

The Raspberry Pi4 is an embedded computer fast enough to run these models in
realtime, if pipelined over all 4 CPU cores, as
```onsets_frames_transcription_realtime.py``` does.
This opens up exciting possibilities of using close-to-realtime transcription
even for small embedded devices and thus interesting potential applications
for musicians.

Unfortunately some dependencies of the current full Magenta codebase make
installation on Raspberry Pi tricky (e.g. librosa), however it is possible
to get this  realtime example running in the following way
(instead of installing the full Magenta environment).

1) Install and set up a fresh Raspbian on an SD card, using the latest
Rapbian Buster from here: https://www.raspberrypi.org/downloads/raspbian/
Specifically we tested with the 2019-09-26 release.

2) Install a standalone version of tflite.
Download a .whl file from <https://www.tensorflow.org/lite/guide/python>
```
pip3 install https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
```

3) Install the following dependencies:
```
sudo apt-get install -y python3 python3-pyaudio python3-numpy python3-scipy
sudo pip3 install colorama absl-py attrs samplerate
```

4) Run the realtime example
Using a USB mic and the models downloaded earlier (see above) you can now run
the example on Raspberry Pi:

```
python3 onsets_frames_transcription_realtime.py \
  --model_path onsets_frames_wavinput_uni.tflite \
  --sample_rate_hz 48000
```

Note, the last parameter was added because most small USB mics can only record
at 48000 Hz, but our model expects 16000 Hz. This parameter will cause the
correct downsampling to happen.
