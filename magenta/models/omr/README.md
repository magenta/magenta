## Magenta Optical Music Recognition

An experimental optical music recognition engine.

Magenta OMR reads PNG image(s) and outputs [MusicXML](https://www.musicxml.com/)
or a [NoteSequence message](../../protobuf/music.proto). MusicXML is a standard
sheet music interchange format, and `NoteSequence` is used internally for
training generative music models.

### Usage

We currently provide a simple CLI for development.

    cd magenta  # Root directory of repo
    bazel build magenta/models/omr
    # Prints a lengthy Score message with low-level information.
    bazel-bin/magenta/models/omr/omr imslp_backup/0/IMSLP00001-000.png
    # Scans several pages and writes MusicXML to ~/symphony.xml. You might want
    # to take a coffee break.
    bazel-bin/magenta/models/omr/omr \
        --output_type=MusicXML --output=$HOME/symphony.xml \
        'imslp_backup/0/IMSLP00033-*.png'
    # Prints a NoteSequence message with symbolic information, useful as input
    # to other Magenta models.
    bazel-bin/magenta/models/omr/omr --output_type=NoteSequence \
        imslp_backup/0/IMSLP00001-000.png

The OMR CLI will print a `Score` message by default, or
[MusicXML](https://www.musicxml.com/) or a `NoteSequence` message if specified.

Magenta OMR is intended to be run in bulk, and not offer a full UI for
correcting the score. The main entry point will be an Apache Beam pipeline that
processes an entire corpus of images.

### Corpus

A large corpus of public domain music scores is available from
[IMSLP](https://imslp.org), and can be [fetched and converted to
PNGs](scripts/imslp_pdfs_to_pngs.sh).

### Table of Contents

* [Concepts](docs/concepts.md)
* [Engine - Detailed Reference](docs/engine.md)
