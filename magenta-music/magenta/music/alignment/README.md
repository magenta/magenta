# align_fine

WAV/MIDI fine alignment tool as described in
[Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset
](https://goo.gl/magenta/maestro-paper).

This implements dynamic time warping in C++ for speed. It is intended to be
used to align WAV/MIDI pairs that are known to be already close to aligned. To
optimize for this case, DTW distance comparisons are calculated on demand for
only the positions within the specified band radius (.5 seconds by default)
rather than calculating the full distance matrix at startup. This allows for
efficient alignment of long sequences.

Note that this is not a supported part of the main `magenta` pip package and
must be run separately from it.

## Prerequisites

1. Install [Bazel](https://bazel.build).
1. Install the `magenta` pip package.

## Usage

From within this directory:

```
INPUT_DIR=<Directory containing .wav and .midi file pairs to be aligned>
OUTPUT_DIR=<Directory to contain aligned .midi files>
bazel run :align_fine -- --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}"
```
