**next_frame_prediction** is a tool dedicated to creating video by learning to predict
a frame from the previous one, and apply it recursively to an initial frame.

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

### Additional dependencies

This tools need some additional dependencies to run.

You'll need to first install [`ffmpeg`](https://www.ffmpeg.org/download.html).

You will also need to copy a modified version of the [`pix2pix-tensorflow`](https://github.com/affinelayer/pix2pix-tensorflow) library using the following commands:

```bash
curl -LO https://github.com/dh7/pix2pix-tensorflow/archive/0.1.tar.gz
tar -xvf 0.1.tar.gz
```

## Goal

The goal of this tool is to create video frame by frame.
From a first arbitrary frame, the algorithm will predict a plausible next one.
Then from this new frame, the algorithm will predict another one, and repeat that until it creates a full video.

Predicting a plausible next frame for a video is an open problem. Several publications have tried to solved it.
The approach here is less scientific and was inpired by [this tweet](https://twitter.com/quasimondo/status/817382760037945344?lang=fr) from Mario Klingemann:
It predicts the next frame from the previous one only, and it tends to diverge into something unrealistic.

The algorithm in this folder implements a technique to prevent the algorithm from diverging as you can see in [this video](https://youtu.be/lr59AhOPgWQ).

To learn how to predict the next frame, the algorithm needs a video as a source (the training set).
The algorithm splits the video in frames, and asks [pix2pix](https://github.com/yenchenlin/pix2pix-tensorflow) to learn to predict frame n+1 from frame n.
The algorithm then generates some predictions (that diverge) and creates a new training set from it.
The algorithm then refines the prediction by asking pix2pix to learn to predict a real frame from the predicted (divergent) ones.

## Usage

To try you'll need a video.mp4 file as the source and two folders:

```bash
mkdir working_folder
mkdir output_folder
cp <your_video> working_folder/video.mp4
```

Then you'll need to launch the main script from this directory:
```bash
./create_video.sh  \
  $(pwd)/working_folder  \
  20  \
  500 \
  $(pwd)/output_folder
```

This script will extract frames from the video.
Then, it will train pix2pix 20 times and generate a video made of 500 frames at each step.

The script will ask you questions, and the first time you launch it, you should answer yes to all of them.
If you interupt the script and want to restart it, you can answer yes or no to decide which part of the process you want to keep or restart from scratch.

There are some constraints that you may want to change by editing the script if the default values don't give the optimal result.

Feedback welcome.
