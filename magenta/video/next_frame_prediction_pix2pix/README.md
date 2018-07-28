**next_frame_prediction** is a tool dedicated to create video by learning to predict
a frame from the previous one, and apply it recursivly to an initial frame.

If you’d like to learn more about Magenta, check out our [blog](https://magenta.tensorflow.org),
where we post technical details.  You can also join our [discussion
group](https://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

## Getting Started

* [Installation](#installation)
* [Goal](#goal)
* [Usage](#usage)

## Installation

### Magenta working environement

Please setup the Magenta developement environement first, following the main documentation [here](https://github.com/tensorflow/magenta#development-environment).

### Additional dependency

This tools need some additional dependency to run.

In particular you'll need to install:
* sk-video (tested with 1.1.8)
* Pillow (tested with 4.0.0)
* Bazel (tested with 0.14.1 and 0.15.2)

```
pip install sk-video
pip install Pillow
```

## Goal

The goal of this tool is to create video frame by frames.
From a first arbitrary frame, the algorythm will predict a plausible next one.
Then from this new frame, the algorythm will predict another one, and repeat that until it created a full video.

Predicting a plausible next frame for a video is an open problem. Several publications tryed to solved it.
The approach here is less scientific and was inpired by [this tweet](https://twitter.com/quasimondo/status/817382760037945344?lang=fr) from Mario Klingemann:
It predict the next frame from the previous one only, and it tends to diverge to something unrealistic.

The algorythm in this folder implement a technique to prevent the algorythm to diverge as you can see in [this video](https://youtu.be/lr59AhOPgWQ).

To learn how to predict the next frame the algorythm need a video as a source (the trainning set).
The algorythm split the video in frames, and ask [pix2pix](https://github.com/yenchenlin/pix2pix-tensorflow) to learn to precict frame n+1 from frame n.
The algorythm then generate some prediction (that diverge) and create a new trainning set from it.
The algorythm then refine the prediction by asking pix2pix to learn to predict a real frame from the predicted (divergent) ones.

## Usage

To try you'll need a video.mp4 file as the source and two folder

```
mkdir working_folder
mkdir output_folder
cp your_video working_folder/video.mp4
```

Then you'll need to use bazel to launch the main script
```
bazel run //magenta/video/next_frame_prediction_pix2pix:create_video  \
  /absolute_path/working_folder/  \
  20  \
  500 \
  /absolute_path/output_folder/
```

This script will extract frames from the video.
Then will train pix2pix 20 times and generate a video made of 500 frames at each step.

The script will ask you questions, and the first time you launch it,
you should answer yes to all of them.
If you interupt the script and want to restart it, you can answer yes or no
to decide witch part of the process you want to keep or restart from scratch.

There are some const that you may want to change by editing the script if the default values don't give the optimal result.

Feedback welcome.
