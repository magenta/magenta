## OMR Engine Detailed Reference

The OMR engine converts a PNG image to a [NoteSequence message](../../../protobuf/music.proto), which is
interoperable with MIDI and MusicXML.

OMR uses TensorFlow for glyph (symbol) classification, as well as any other
compute-intensive steps. Final processing is done in Python. Evaluation is done
by [OMREngine.run](../engine.py). Glyph classification is configurable by the
`glyph_classifier_fn` argument, and other components of image recognition are
part of the [Structure](../structure/__init__.py).

### Diagram

<img src="engine_diagram.svg">

### API

The entry point for running OMR is [`OMREngine.run()`](../engine.py).
It takes in a list of PNG filenames and outputs a `Score` or `NoteSequence`
message. The `Score` can further be converted to
[MusicXML](../conversions/musicxml.py).

### TensorFlow Graph

For maximum parallelism, all processing is run in the same TensorFlow graph. The
graph is run by `OMREngine._get_page()`.  This also evaluates the `Structure`
in the same graph. `Structure` is a wrapper for detectors which extract
information from the image, including
[staves](../staves/hough.py), [vertical lines](../structure/verticals.py), and
[note beams](../structure/beams.py).

#### Structure

[Structure](../structure/__init__.py) holds the structural
elements that need to be evaluated for OMR, but does not do any symbol
recognition. The structure encompasses staves, beams, and vertical lines, and
may contain more elements in the future (e.g. full connected component analysis)
which can be used to detect more elements (e.g. note dots). Structure detection
is currently simple computer vision rather than ML, but it can easily be swapped
out with a different TensorFlow model.

##### Staffline Distance Estimation

We estimate the [staffline
distance(s)](../staves/staffline_distance.py) of
the entire image. There may be staves with multiple different sizes for
different parts on a single page, but there should be just a few possible
staffline distance values.

##### Staff Detection

Concrete subclasses of
[BaseStaffDetector](../staves/base.py) take in the image
and produce:

*   `staves`: Tensor of shape `(num_staves, num_points, 2)`. Coordinates of the
    staff center line (third line on the staff).
*   `staffline_distance`: Vector of the estimated staffline distance (distance
    between consecutive staff lines) for each staff.
*   `staffline_thickness`: Scalar thickness of staff lines. Assumed to be the
    same for all staves.
*   `staves_interpolated_y`: Tensor of shape `(num_staves, width)`. For each
    staff and column of the image, outputs the interpolated y position of the
    staff center line.

##### Staff Removal

[StaffRemover](../staves/removal.py) takes in the image
and staves, and outputs `remove_staves` which is the image with the staff lines
erased. This is useful so that [glyphs](concepts.md) look the same whether they
are centered on a staff line or the space between lines. It is also used within
the structure, by beam detection.

##### Beam Detection

[Beams](../structure/beams.py) are currently detected from
connected components on an
[eroded](https://en.wikipedia.org/wiki/Mathematical_morphology#Binary_morphology)
staves-removed image. These are attached to notes by a `BeamProcessor`.

##### Vertical Line Detection

[ColumnBasedVerticals](../structure/verticals.py) detects
all vertical lines in the image. These will later be used as either stems or
barlines.

#### Glyph Classification: 1-D Convolutional Model

We also run a glyph classifier as part of the TensorFlow graph, which outputs
predictions.

##### Staffline Extraction

Glyphs are considered to lie on a black staff line, or halfway between staff
lines. For OMR, extracted stafflines are slices of the image that are either
centered on an staff line, or halfway between staff lines. The line that the
extracted staffline lies on may just be referred to a staffline, or a [y
position](https://github.com/ringw/magenta/blob/a4b203c5185d934fcffa13912a6702058e6c4e68/magenta/models/omr/protobuf/musicscore.proto#L70) of the staff.

[StafflineExtractor](../staves/staffline_extractor.py)
extracts these vertical slices of the image, and scales their height to a
constant value (currently, 18 pixels tall). `StaffRemover` is used so that all
extracted stafflines should look similar.

##### Glyph Classification

[Glyphs](concepts.md) are classified on small, horizontal slices (currently, 15
pixels wide) of the extracted staffline, a 1D convolutional model.

A [GlyphClassifier](../glyphs/base.py) outputs a Tensor
`staffline_predictions` of shape `(num_staves, num_stafflines, width)`. The
values are for the `Glyph.Type` enum. Value 0 (UNKNOWN_TYPE) is not used;
value 1 (NONE) corresponds to no glyph.

### Post-Processing

#### Page Construction

OMR processing operates on [Page
protos](../protobuf/musicscore.proto). The Page is first
constructed by `BaseGlyphClassifier.get_page`, which populates the glyphs on
each staff. Staff location information is then added by `StaffProcessor`.

Single `Glyph`s are created from consecutive runs in `staffline_predictions`
that are classified as the same glyph type.

Additional processors modify the `Page` in place, usually adding information
from the `Structure`. Each page is run through
[`page_processors.process()`](../page_processors.py), and
then the score (containing all pages) is run through
[`score_processors.process()`](../score_processors.py).

#### Stem Detection

[Stems](../structure/stems.py) finds stem candidates from
the vertical lines, and adds a `Stem` to notehead `Glyph`s if the closest stem
is close enough to the expected position. The `ScoreReader` considers multiple
noteheads with identical `Stem`s as a single chord. Stems will also be used as a
negative signal to avoid detecting barlines in the same area.

#### Beam Processing

Beams from the `Structure` that are close enough to a stem are added to one or
more notes by [BeamProcessor](../structure/beam_processor.py).

#### Barlines

[Barlines](../structure/barlines.py) are detected from the
verticals if they have not already been used as a stem.

#### Score Reading

The [ScoreReader](../score/reader.py) is the only score
processor. It can potentially use state that lasts across multiple pages, such
as the current time in the score, which needs to persist for the entire score.

Staves are scanned from left to right for glyphs. The `ScoreReader` manages a
hierarchy of state, from the global `ScoreState` to the `MeasureState`, holding
local state such as accidentals. Based on the preceding `Glyph`s, each notehead
`Glyph` gets assigned a
[`Note`](https://github.com/tensorflow/magenta/blob/2e7bc85eb30e100733ea34c0ecdd080a4cec1c1f/magenta/protobuf/music.proto#L93).

Afterwards, the `Score` can be converted to a `NoteSequence` (just pulling out
all of the `Note`s) or
[MusicXML](../conversions/musicxml.py).
