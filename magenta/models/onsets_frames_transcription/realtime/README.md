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

### Demo

First, set up your [Magenta environment](/README.md).

Then, run the demo:

```
onsets_frames_transcription_realtime \
  --model_path /path/to/onsets_frames_wavinput.tflite
```
