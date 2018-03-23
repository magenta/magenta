/**
 * @fileoverview VexFlow random labeled data generator.
 *
 * Outputs a JSON list of dicts with "svg" (XML text SVG image) and "page" (text
 * format Staff proto holding the glyph coordinates).
 */

const ArgumentParser = require('argparse').ArgumentParser;
const jsdom = require('jsdom');
const Random = require('random-js');
const Vex = require('vexflow');
const VF = Vex.Flow;

const parser = new ArgumentParser();
parser.addArgument(
    ['--random_seeds'],
    {help: 'Generate a labeled image for each comma-separated random seed.'});
const args = parser.parseArgs();


/**
 * @param {?object} value Any value
 * @param {string=} opt_message The message, if value is null or undefined
 * @return {!object} The non-null value
 * @throws {ValueError} if value is null or undefined
 */
function checkNotNull(value, opt_message) {
  // undefined == null too.
  if (value == null) {
    throw ValueError(opt_message);
  }
  return value;
}


/**
 * VexFlow line numbers start at the first ledger line below the staff, and
 * increment by a half for each note. OMR y positions start at the third staff
 * line, and increment by 1 for each note.
 * @param {number} line The VexFlow line number.
 * @return {number} An integer y position.
 */
function vexflowLineToOMR(line) {
  return (line - 3) * 2;
}


/**
 * Converts an absolute Y coordinate on a staff to an OMR y position.
 * @param {!Vex.Flow.Stave} stave The VexFlow stave.
 * @param {number} y The y coordinate, in pixels.
 * @return {number} An OMR y position.
 */
function absoluteYToOMR(stave, y) {
  const staff_center_y = stave.getYForLine(2);
  return Math.round((staff_center_y - y) * 2 / stave.space(1));
}


let allGlyphs, staffCenterLine, stafflineDistance;
/** Resets the dumped VexFlow information before starting a new page. */
function resetPageState() {
  allGlyphs = [];
  staffCenterLine = null;
  stafflineDistance = null;
}
resetPageState();


const CLEF_LINE_FOR_OMR = {
  // Treble or "G" clef is centered 2 spaces below the center line (on G).
  'treble': -2,
  // Bass or "F" clef is centered 2 spaces above the center line (on F).
  'bass': +2
};
const drawClef = checkNotNull(VF.Clef.prototype.draw, 'Clef.draw');
/** Draws the clef and dumps its position to allGlyphs. */
VF.Clef.prototype.draw = function() {
  if (this.type == 'treble' || this.type == 'bass') {
    const x = Math.round(this.getX() + this.getWidth() / 2);
    allGlyphs.push(`glyph {
type: CLEF_${this.type.toUpperCase()}
x: ${x}
y_position: ${CLEF_LINE_FOR_OMR[this.type]}
}`);
  }
  drawClef.apply(this, arguments);
};


const drawStave = checkNotNull(VF.Stave.prototype.draw, 'Stave.draw');
/** Dumps the staff information. */
VF.Stave.prototype.draw = function() {
  stafflineDistance = this.space(1);
  const y = this.getYForLine(2);
  const x0 = this.getX();
  const x1 = this.getX() + this.getWidth();
  staffCenterLine = `center_line {
  x: ${x0}
  y: ${y}
}
center_line {
  x: ${x1}
  y: ${y}
}
`;
  drawStave.apply(this, arguments);
};

const drawNotehead = checkNotNull(VF.NoteHead.prototype.draw, 'Notehead.draw');
/** Draws the notehead and dumps its position to allGlyphs. */
VF.NoteHead.prototype.draw = function() {
  // The notehead x seems to be the left end.
  const x = Math.round(this.getAbsoluteX() + this.getWidth() / 2);
  const y_position = vexflowLineToOMR(this.getLine());
  allGlyphs.push(`glyph {
# TODO(ringwalt): NOTEHEAD_FILLED vs NOTEHEAD_EMPTY.
type: NOTEHEAD_FILLED
x: ${x}
y_position: ${y_position}
}`);
  drawNotehead.apply(this, arguments);
};

const ACCIDENTAL_TYPES = {
  'b': 'FLAT',
  '#': 'SHARP',
  'n': 'NATURAL'
};
const drawAccidental =
    checkNotNull(VF.Accidental.prototype.draw, 'Accidental.draw');
/** Draws the accidental and dumps its position to allGlyphs. */
VF.Accidental.prototype.draw = function() {
  if (this.type in ACCIDENTAL_TYPES) {
    const note_start = this.note.getModifierStartXY(this.position, this.index);
    // The modifier x (note_start.x + this.x_shift) seems to be the right end of
    // the glyph.
    const x = Math.round(note_start.x + this.x_shift - this.getWidth() / 2);
    const y = note_start.y + this.y_shift;
    const y_position = absoluteYToOMR(this.note.getStave(), y);
    allGlyphs.push(`glyph {
type: ${ACCIDENTAL_TYPES[this.type]}
x: ${x}
y_position: ${y_position}
}`);
  }
  drawAccidental.apply(this, arguments);
};


/**
 * @param {!Random} random The random generator
 * @param {!object<number>} probs Map from key to probability. The probability
 *     values must sum to 1.
 * @return {!object} The sampled key from probs.
 */
function discreteSample(random, probs) {
  let cumulativeProb = 0;
  const randomUniform = random.real(0, 1);
  for (let key of Object.keys(probs)) {
    if (randomUniform < cumulativeProb + probs[key]) {
      return key;
    }
    cumulativeProb += probs[key];
  }
  throw ValueError('Probabilities sum to ' + cumulativeProb);
}


const PROB_LEDGER_NOTE = 0.1;
const PROB_MODIFIERS = {
  '#': 0.25,
  'b': 0.25,
  '##': 0.02,
  'bb': 0.02,
  'n': 0.15,
  '': 0.31,
};
class Clef {
  /**
   * Samples a random note to display for the clef.
   * @param {!Random} random The random generator
   * @return {!string} The note name, with accidental.
   */
  genNote(random) {
    const modifier = discreteSample(random, PROB_MODIFIERS);
    return random.pick(this.baseNotes_()).replace('(?=[0-9])', modifier);
  }

  // TODO(ringwalt): Why does the bass clef render notes in the same positions
  // as a treble clef? Fix this and add a different range of notes for bass.
  /**
   * @return {!array<!string>} The base note names (without accidentals) for
   * notes that lie on the staff, or are within 2 ledger lines.
   * @private
   */
  baseNotes_() {
    return [
      'A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5',
      'F5', 'G5', 'A5', 'B5', 'C6'
    ];
  }
}


class TrebleClef extends Clef {
  /** @return {!string} the name used by VexFlow for the clef. */
  name() {
    return 'treble';
  }
}


class BassClef extends Clef {
  /** @return {!string} the name used by VexFlow for the clef. */
  name() {
    return 'bass';
  }
}


const CLEFS = [new TrebleClef(), new BassClef()];


if (!args.random_seeds) {
  throw Error('--random_seeds is required');
}
const seedStrings = args.random_seeds.split(',');
const seeds = [];
seedStrings.forEach(function(seedString) {
  const seed = parseInt(seedString, 10);
  if (isNaN(seed)) {
    throw Error('Seed is not an integer: ' + seedString);
  }
  seeds.push(seed);
});


jsdom.env({
  html: '<div id="vexflow-div" />',
  done: function(errors, window) {
    if (errors) {
      throw Error('node-jsdom failed: ' + errors);
    }
    global.window = window;
    global.document = window.document;

    const vf = new Vex.Flow.Factory(
        {renderer: {elementId: 'vexflow-div', width: 500, height: 200}});
    const staveConstructor = vf.Stave;
    // TODO(ringwalt): Support passing Vex.Flow.Stave options through addStave()
    vf.Stave = function(params) {
      const paramsCopy = {};
      Object.assign(paramsCopy, params);
      const options = {fill_style: '#000000'};
      Object.assign(options, paramsCopy.options);
      paramsCopy.options = options;
      return staveConstructor.apply(this, [paramsCopy]);
    };

    pages = [];
    seeds.forEach(function(seed) {
      const random = new Random(Random.engines.mt19937().seed(seed));
      const clef = random.pick(CLEFS);

      const score = vf.EasyScore();
      const system = vf.System();

      const notes = [];
      for (let i = 0; i < 4; i++) {
        notes.push(clef.genNote(random));
      }
      // TODO(ringwalt): Random durations.
      notes[0] = notes[0] + '/q';

      system
          .addStave({
            voices: [score.voice(score.notes(notes.join(', ')))],
          })
          .addClef(clef.name())
          .addTimeSignature('4/4');

      vf.draw();

      let page_message = `staffline_distance: ${stafflineDistance}
${staffCenterLine}
`;
      allGlyphs.forEach(function(glyph) {
        page_message = page_message + glyph;
      });
      pages.push({
        'svg': document.getElementById('vexflow-div').innerHTML,
        'page': page_message
      });

      resetPageState();
    });

    process.stdout.write(JSON.stringify(pages));
  }
});
