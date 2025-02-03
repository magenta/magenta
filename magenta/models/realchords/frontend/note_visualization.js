/**
 * @fileoverview Rendering for piano and note scrolling.
 *
 * @author Yusong Wu (yusongw@google.com)
 * https://github.com/lukewys/PianorollVis.js
 *
 * @author Alex Scarlatos (scarlatos@google.com)
 * @suppress {visibility}
 */

const CONSTANTS = {
  COLORS: {
    red: '#EE2B29',
    orange: '#ff9800',
    yellow: '#ffff00',
    green: '#c6ff00',
    cyan: '#00e5ff',
    blue: '#2979ff',
    lightBlue: '#7daaf5',
    purple: '#651fff',
    meta: '#d500f9'
  },
  NOTES_PER_OCTAVE: 12,
  WHITE_NOTES_PER_OCTAVE: 7,
  LOWEST_PIANO_KEY_MIDI_NOTE: 21,
  REFRESH_RATE: 30,
};

/** @enum {string} */
const computerKeys = [
  'a', 'w', 's', 'e', 'd', 'f', 't', 'g', 'y', 'h', 'u', 'j', 'k', 'o', 'l',
  'p', ';', '\''
];
/** @enum {number} */
const blackNoteIndexes = [1, 3, 6, 8, 10];
/** @enum {number} */
const whiteWideNoteIndexes = [2, 7, 9];        // two black key neighbors
/** @enum {number} */
const whiteNarrowNoteIndexes = [0, 4, 5, 11];  // one black key neighbor

class BasePiano {
  /** Set base parameters for piano renderer class */
  constructor(svg, width = -1, height = -1, x = 1, y = 1, lowestC = 1) {
    this.config = {
      whiteNoteWidth: undefined,
      blackNoteWidth: undefined,
      whiteNoteHeight: undefined,
      blackNoteHeight: undefined
    };
    this.computerKeyOctave = 4;
    this.octaves = 7;
    this.lowestC = lowestC;
    this.svg = svg;
    this.svgNS = 'http://www.w3.org/2000/svg';
    if (width === -1) {
      this.width = window.innerWidth;
    } else {
      this.width = width;
    }
    if (height === -1) {
      this.height = window.innerHeight;
    } else {
      this.height = height;
    }
    this.x = x;
    this.y = y;
  }

  /** Set the octave the computer keyboard starts on */
  setComputerKeyOctave(computerKeyOctave) {
    this.computerKeyOctave = computerKeyOctave;
    this.draw();
    this.drawBackgroundGrid();
  }

  /**
   * Highlight a note on the piano
   * @param {number} note
   * @param {string} color
   * @return {Object!}
   */
  highlightNote(note, color) {
    // Show the note on the piano roll.
    const rect = this.findNoteRect(note);
    // console.log(rect);
    if (!rect) {
      console.log('couldnt find a rect for note', note);
      return;
    }
    rect.setAttribute('active', true);
    rect.setAttribute('class', color);
    return rect;
  }

  /**
   * Return the rect object associated with the given note
   * @param {number} note
   * @return {Object!}
   */
  findNoteRect(note) {
    return this.svg.querySelector(`rect[data-index="${note}"]`);
  }

  /** Clear all active and class in rect */
  clearNote(rect) {
    rect.removeAttribute('active');
    rect.removeAttribute('class');
  }

  /**
   * Make single rect element
   * (x,y) is the coordinate of the upper left corner of the rectangle
   * @param {number} index
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {number} noteSize
   * @param {string} fill
   * @param {string} stroke
   * @return {Object!}
   */
  makeRect(index, x, y, w, h, noteSize, fill, stroke) {
    const rect = document.createElementNS(this.svgNS, 'rect');
    rect.setAttribute('data-index', index);
    rect.setAttribute('x', x);
    rect.setAttribute('y', y);
    if (this.octaves > 6) {
      rect.setAttribute('x_position', index * noteSize);
      rect.setAttribute('y_position', this.height - ((index + 1) * noteSize));
    } else {
      rect.setAttribute(
          'x_position',
          (index - 3 - CONSTANTS.NOTES_PER_OCTAVE * (this.lowestC - 1)) *
              noteSize);
      rect.setAttribute(
          'y_position',
          this.height -
              (index - 2 - CONSTANTS.NOTES_PER_OCTAVE * (this.lowestC - 1)) *
                  noteSize);
    }
    rect.setAttribute('width', w);
    rect.setAttribute('height', h);
    rect.setAttribute('noteSize', noteSize);
    rect.setAttribute('fill', fill);
    if (stroke) {
      rect.setAttribute('stroke', stroke);
      rect.setAttribute('stroke-width', '2px');
    }
    this.svg.appendChild(rect);
    return rect;
  }

  /**
   * Make single grid line
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {string} fill
   * @param {string} stroke
   * @return {Object!}
   */
  makeGrid(x, y, w, h, fill, stroke) {
    const rect = document.createElementNS(this.svgNS, 'rect');
    rect.setAttribute('x', x);
    rect.setAttribute('y', y);
    rect.setAttribute('width', w);
    rect.setAttribute('height', h);
    rect.setAttribute('fill', fill);
    if (stroke) {
      rect.setAttribute('stroke', stroke);
      rect.setAttribute('stroke-width', '1px');
    }
    this.svg.appendChild(rect);
    return rect;
  }

  /**
   * Make text element
   * @param {number} x
   * @param {number} y
   * @param {number} note
   * @param {string} fill
   * @param {number} fontSize
   * @param {string} fontFamily
   * @param {string} pointerEvents
   * @return {Object!}
   */
  makeText(x, y, note, fill, fontSize, fontFamily, pointerEvents) {
    const text = document.createElementNS(this.svgNS, 'text');
    text.setAttribute('x', x);
    text.setAttribute('y', y);
    text.setAttribute('fill', fill);
    text.setAttribute('font-size', fontSize);
    text.setAttribute('font-family', fontFamily);
    text.setAttribute('pointer-events', pointerEvents);
    text.textContent = note;

    this.svg.appendChild(text);
    return text;
  }

  /** Callback for screen resizing */
  resize() {
    throw new Error('Implement Required');
  }

  /** Callback for piano rendering */
  draw() {
    throw new Error('Implement Required');
  }

  /** Clear all content in svg */
  clearSVG() {
    while (this.svg.firstChild) {
      this.svg.removeChild(this.svg.firstChild);
    }
  }
}

class HorizontalPiano extends BasePiano {
  /** Set config for horizontal piano */
  constructor(svg, width = -1, height = -1, x = 1, y = 1, lowestC = 1) {
    super(svg, width, height, x, y, lowestC);
    this.config = {
      whiteNoteWidth: 100,
      blackNoteWidth: 2 * 100 / 3,
      whiteWideNoteHeight: 20,
      whiteNarrowNoteHeight: 20,
      whiteHighestCHeight: 20,
      blackNoteHeight: 20
    };
  }

  /** Redraw piano after resizing window */
  resize(totalWhiteNotes) {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    if (this.octaves > 6) {
      this.config.blackNoteHeight =
          this.height / (this.octaves * CONSTANTS.NOTES_PER_OCTAVE + 4);
    } else {
      this.config.blackNoteHeight =
          this.height / (this.octaves * CONSTANTS.NOTES_PER_OCTAVE);
    }
    this.config.whiteNarrowNoteHeight = this.config.blackNoteHeight * 3 / 2;
    this.config.whiteWideNoteHeight = this.config.blackNoteHeight * 2;
    this.config.whiteHighestCNoteHeight = this.config.blackNoteHeight;
    console.log('totalWhiteNotes', totalWhiteNotes);
    this.svg.setAttribute('width', this.width);
    this.svg.setAttribute('height', this.height);
    // position svg to this.x and this.y
    this.svg.style.left = `${this.x}px`;
    this.svg.style.top = `${this.y}px`;
    this.svg.style.position = 'absolute';
  }

  /** Draw background grid */
  drawBackgroundGrid() {
    let x = this.config.whiteNoteWidth;
    let y = this.height - this.config.blackNoteHeight;

    const blackNoteIndexes = [1, 3, 6, 8, 10];

    // draw bonus notes
    if (this.octaves > 6) {
      this.makeGrid(
          x, y, this.width - x, this.config.blackNoteHeight, '#474747',
          '#5C5C5C');
      this.makeGrid(
          x, y - 2 * this.config.blackNoteHeight, this.width - x,
          this.config.blackNoteHeight, '#474747', '#5C5C5C');
      y = this.height - 4 * this.config.blackNoteHeight;
    } else {
      // Starting 3 semitones up on small screens (on a C), and a whole octave
      // up.
      y = this.height - this.config.blackNoteHeight;
    }

    for (let o = 0; o < this.octaves; o++) {
      for (let i = 0; i < CONSTANTS.NOTES_PER_OCTAVE; i++) {
        if (blackNoteIndexes.indexOf(i) === -1) {
          this.makeGrid(
              x, y, this.width - x, this.config.blackNoteHeight, '#474747',
              '#5C5C5C');
        }
        y -= this.config.blackNoteHeight;
      }
    }

    if (this.octaves > 6) {
      // And an extra C at the end (if we're using all the octaves);
      this.makeGrid(
          x, y, this.width - x, this.config.blackNoteHeight, '#474747',
          '#5C5C5C');

      // Now draw all the black notes, so that they sit on top.
      // Pianos start on an A:
      this.makeGrid(
          x, this.height - 2 * this.config.blackNoteHeight, this.width - x,
          this.config.blackNoteHeight, '#303030');
      y = this.height - 4 * this.config.blackNoteHeight;
    } else {
      // Starting 3 semitones up on small screens (on a C), and a whole octave
      // up.
      y = this.height - this.config.blackNoteHeight;
    }

    for (let o = 0; o < this.octaves; o++) {
      for (let i = 0; i < CONSTANTS.NOTES_PER_OCTAVE; i++) {
        if (blackNoteIndexes.indexOf(i) !== -1) {
          this.makeGrid(
              x, y, this.width - x, this.config.blackNoteHeight, '#303030');
        }
        y -= this.config.blackNoteHeight;
      }
    }
  }

  /**
   * Draw all black notes for one octave
   * @param {number} x
   * @param {number} y
   * @param {number} index
   * @param {number} halfABlackNote
   */
  drawBlackNotesPerOctave(x, y, index, halfABlackNote) {
    // C#
    this.makeRect(
        index, x, y - this.config.whiteNarrowNoteHeight - halfABlackNote,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteHeight, 'black');
    index += 2;
    y -= this.config.whiteNarrowNoteHeight;
    // D#
    this.makeRect(
        index, x, y - this.config.whiteWideNoteHeight - halfABlackNote,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteHeight, 'black');
    index += 3;
    y -= this.config.whiteWideNoteHeight + this.config.whiteNarrowNoteHeight;
    // F#
    this.makeRect(
        index, x, y - this.config.whiteNarrowNoteHeight - halfABlackNote,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteHeight, 'black');
    index += 2;
    y -= this.config.whiteNarrowNoteHeight;
    // G#
    this.makeRect(
        index, x, y - this.config.whiteWideNoteHeight - halfABlackNote,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteHeight, 'black');
    index += 2;
    y -= this.config.whiteWideNoteHeight;
    // A#
    this.makeRect(
        index, x, y - this.config.whiteWideNoteHeight - halfABlackNote,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteHeight, 'black');
  }

  /** Draw all piano notes */
  draw() {
    this.clearSVG();
    const halfABlackNote = this.config.blackNoteHeight / 2;
    let x = 0;
    let y = this.height - this.config.whiteNarrowNoteHeight;
    let index = 0;

    const noteList =
        ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

    // First draw all the white notes.
    // Pianos start on an A (if we're using all the octaves);
    if (this.octaves > 6) {
      this.makeRect(
          0, x, y, this.config.whiteNoteWidth,
          this.config.whiteNarrowNoteHeight, this.config.blackNoteHeight,
          'white', '#141E30');
      this.makeRect(
          2, x, y - this.config.whiteNarrowNoteHeight,
          this.config.whiteNoteWidth, this.config.whiteNarrowNoteHeight,
          this.config.blackNoteHeight, 'white', '#141E30');
      index = 3;
      y = this.height - 2 * this.config.whiteNarrowNoteHeight;
    } else {
      // Starting 3 semitones up on small screens (on a C), and a whole octave
      // up.
      index = 3 + (this.lowestC - 1) * CONSTANTS.NOTES_PER_OCTAVE;
      y = this.height;
    }

    // Draw the white notes.
    for (let o = 0; o < this.octaves; o++) {
      for (let i = 0; i < CONSTANTS.NOTES_PER_OCTAVE; i++) {
        if (whiteWideNoteIndexes.indexOf(i) !== -1) {
          y -= this.config.whiteWideNoteHeight;
          this.makeRect(
              index, x, y, this.config.whiteNoteWidth,
              this.config.whiteWideNoteHeight, this.config.blackNoteHeight,
              'white', '#141E30');
        } else if (whiteNarrowNoteIndexes.indexOf(i) !== -1) {
          y -= this.config.whiteNarrowNoteHeight;
          this.makeRect(
              index, x, y, this.config.whiteNoteWidth,
              this.config.whiteNarrowNoteHeight, this.config.blackNoteHeight,
              'white', '#141E30');
        }
        index++;
      }
      // label the C of each octave.
      let note = noteList[0] + (o + this.lowestC);
      let fontSize = 14;
      let fontFamily = 'cursive';
      let pointerEvents = 'none';
      this.makeText(
          this.config.whiteNoteWidth - 1.6 * fontSize,
          y + 12 * this.config.blackNoteHeight -
              (this.config.whiteNarrowNoteHeight - fontSize) / 2 - 1.5,
          note, 'black', fontSize, fontFamily, pointerEvents);
    }

    if (this.octaves > 6) {
      // Add an extra C at the end (if we're using all the octaves);
      y -= this.config.whiteHighestCNoteHeight;
      this.makeRect(
          index, x, y, this.config.whiteNoteWidth,
          this.config.whiteHighestCNoteHeight, this.config.blackNoteHeight,
          'white', '#141E30');

      // Now draw all the black notes, so that they sit on top.
      // Pianos start on an A:
      this.makeRect(
          1, x,
          this.height - (this.config.whiteNarrowNoteHeight + halfABlackNote),
          this.config.blackNoteWidth, this.config.blackNoteHeight,
          this.config.blackNoteHeight, 'black');
      index = 4;
      y = this.height - 2 * this.config.whiteNarrowNoteHeight;
    } else {
      // Starting 3 semitones up on small screens (on a C), and a whole octave
      // up.
      index = 4 + (this.lowestC - 1) * CONSTANTS.NOTES_PER_OCTAVE;
      y = this.height;
    }

    // Draw the black notes.
    for (let o = 0; o < this.octaves; o++) {
      this.drawBlackNotesPerOctave(x, y, index, halfABlackNote);
      y -= CONSTANTS.NOTES_PER_OCTAVE * this.config.blackNoteHeight;
      index += CONSTANTS.NOTES_PER_OCTAVE;
    }
  }

  /**
   * Set number of bars for cycle for when notes render in cycle
   * @param {number} measureNum
   */
  setCycle(measureNum) {
    // duration: bars
    let offset = this.config.whiteNoteWidth;
    let beatWidth = (this.width - offset) / measureNum / 4;
    // draw a vertical line for each beat
    for (let i = 0; i < measureNum * 4; i++) {
      let line = document.createElementNS(this.svgNS, 'line');
      line.setAttribute('x1', `${beatWidth * i + offset}px`);
      line.setAttribute('y1', '0px');
      line.setAttribute('x2', `${beatWidth * i + offset}px`);
      line.setAttribute('y2', `${this.height}px`);
      line.setAttribute('stroke', '#5C5C5C');
      line.setAttribute('stroke-width', '1px');
      this.svg.appendChild(line);
    }
  }
}

class VerticalPiano extends BasePiano {
  /** Set config for vertical piano */
  constructor(
      svg, width = -1, height = -1, x = 1, y = 1, lowestC = 1,
      atBottom = true) {
    super(svg, width, height, x, y, lowestC);
    this.atBottom = atBottom;
    this.config = {
      whiteWideNoteWidth: 20,
      whiteNarrowNoteWidth: 20,
      whiteHighestCWidth: 20,
      blackNoteWidth: 20,
      whiteNoteHeight: 100,
      blackNoteHeight: 2 * 100 / 3
    };
  }

  /** Redraw piano after resizing window */
  resize(totalWhiteNotes) {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    if (this.octaves > 6) {
      this.config.blackNoteWidth =
          this.width / (this.octaves * CONSTANTS.NOTES_PER_OCTAVE + 4);
    } else {
      this.config.blackNoteWidth =
          this.width / (this.octaves * CONSTANTS.NOTES_PER_OCTAVE);
    }
    this.config.whiteNarrowNoteWidth = this.config.blackNoteWidth * 3 / 2;
    this.config.whiteWideNoteWidth = this.config.blackNoteWidth * 2;
    this.config.whiteHighestCNoteWidth = this.config.blackNoteWidth;
    this.svg.setAttribute('width', this.width);
    this.svg.setAttribute('height', this.height);
    // position svg to this.x and this.y
    this.svg.style.left = `${this.x}px`;
    this.svg.style.top = `${this.y}px`;
    this.svg.style.position = 'absolute';
  }

  /**
   * Render a single piano key, render computer key if in range
   * @param {number} index
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {number} noteSize
   * @param {string} fill
   * @param {string} stroke
   */
  makeKey(index, x, y, w, h, noteSize, fill, stroke) {
    const fontSize = 14;
    this.makeRect(index, x, y, w, h, noteSize, fill, stroke);
    const computerKeyStartIdx = (this.computerKeyOctave - 1) * 12 + 3;
    if (computerKeyStartIdx <= index <
        computerKeyStartIdx + computerKeys.length) {
      const keyIdx = index - computerKeyStartIdx;
      const isBlack = blackNoteIndexes.includes(keyIdx % 12);
      const noteWidth = isBlack ? this.config.blackNoteWidth :
          whiteNarrowNoteIndexes.includes(keyIdx % 12) ?
                                  this.config.whiteNarrowNoteWidth :
                                  this.config.whiteWideNoteWidth;
      const noteHeight =
          isBlack ? this.config.blackNoteHeight : this.config.whiteNoteHeight;
      this.makeText(
          x + noteWidth / 2, y + noteHeight - fontSize, computerKeys[keyIdx],
          isBlack ? 'white' : 'black', fontSize, 'cursive', 'none');
    }
  }

  /** Draw background grid */
  drawBackgroundGrid() {
    let x = 0;
    let y = this.atBottom ? 0 : this.config.whiteNoteHeight;
    let height = this.height - this.config.whiteNoteHeight;

    const blackNoteIndexes = [1, 3, 6, 8, 10];

    // draw bonus notes
    if (this.octaves > 6) {
      this.makeGrid(
          x, y, this.config.blackNoteWidth, height, '#474747', '#5C5C5C');
      this.makeGrid(
          x + 2 * this.config.blackNoteWidth, y, this.config.blackNoteWidth,
          height, '#474747', '#5C5C5C');
      x = 3 * this.config.blackNoteWidth;
    } else {
      // Starting 3 semitones up on small screens (on a C).
      x = 0;
    }

    for (let o = 0; o < this.octaves; o++) {
      for (let i = 0; i < CONSTANTS.NOTES_PER_OCTAVE; i++) {
        if (blackNoteIndexes.indexOf(i) === -1) {
          this.makeGrid(
              x, y, this.config.blackNoteWidth, height, '#474747', '#5C5C5C');
        }
        x += this.config.blackNoteWidth;
      }
    }

    if (this.octaves > 6) {
      // And an extra C at the end (if we're using all the octaves);
      this.makeGrid(
          x, y, this.config.blackNoteWidth, height, '#474747', '#5C5C5C');

      // Now draw all the black notes, so that they sit on top.
      // Pianos start on an A:
      this.makeGrid(
          this.config.blackNoteWidth, y, this.config.blackNoteWidth, height,
          '#303030');
      x = 3 * this.config.blackNoteWidth;
    } else {
      // Starting 3 semitones up on small screens (on a C), and a whole octave
      // up.
      x = 0;
    }

    for (let o = 0; o < this.octaves; o++) {
      for (let i = 0; i < CONSTANTS.NOTES_PER_OCTAVE; i++) {
        if (blackNoteIndexes.indexOf(i) !== -1) {
          this.makeGrid(x, y, this.config.blackNoteWidth, height, '#303030');
        }
        x += this.config.blackNoteWidth;
      }
    }
  }

  /**
   * Draw all black notes for one octave
   * @param {number} x
   * @param {number} y
   * @param {number} index
   * @param {number} halfABlackNote
   */
  drawBlackNotesPerOctave(x, y, index, halfABlackNote) {
    // C#
    this.makeKey(
        index, x + this.config.whiteNarrowNoteWidth - halfABlackNote, y,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteWidth, 'black');
    index += 2;
    x += this.config.whiteNarrowNoteWidth;
    // D#
    this.makeKey(
        index, x + this.config.whiteWideNoteWidth - halfABlackNote, y,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteWidth, 'black');
    index += 3;
    x += this.config.whiteWideNoteWidth + this.config.whiteNarrowNoteWidth;
    // F#
    this.makeKey(
        index, x + this.config.whiteNarrowNoteWidth - halfABlackNote, y,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteWidth, 'black');
    index += 2;
    x += this.config.whiteNarrowNoteWidth;
    // G#
    this.makeKey(
        index, x + this.config.whiteWideNoteWidth - halfABlackNote, y,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteWidth, 'black');
    index += 2;
    x += this.config.whiteWideNoteWidth;
    // A#
    this.makeKey(
        index, x + this.config.whiteWideNoteWidth - halfABlackNote, y,
        this.config.blackNoteWidth, this.config.blackNoteHeight,
        this.config.blackNoteWidth, 'black');
  }

  /** Draw all piano notes */
  draw() {
    this.clearSVG();
    const halfABlackNote = this.config.blackNoteWidth / 2;
    let x = 0;
    let y = this.atBottom ? this.height - this.config.whiteNoteHeight : 0;
    let index = 0;

    // First draw all the white notes.
    // Pianos start on an A (if we're using all the octaves);
    if (this.octaves > 6) {
      this.makeKey(
          0, x, y, this.config.whiteNarrowNoteWidth,
          this.config.whiteNoteHeight, this.config.blackNoteWidth, 'white',
          '#141E30');
      this.makeKey(
          2, this.config.whiteNarrowNoteWidth, y,
          this.config.whiteNarrowNoteWidth, this.config.whiteNoteHeight,
          this.config.blackNoteWidth, 'white', '#141E30');
      index = 3;
      x += 2 * this.config.whiteNarrowNoteWidth;
    } else {
      // Starting 3 semitones up on small screens (on a C), and a whole octave
      // up. (draw the first C from the top, skipping A, A# and B)
      index = 3 + (this.lowestC - 1) * CONSTANTS.NOTES_PER_OCTAVE;
      x = 0;
    }

    // Draw the white notes.
    for (let o = 0; o < this.octaves; o++) {
      for (let i = 0; i < CONSTANTS.NOTES_PER_OCTAVE; i++) {
        if (whiteWideNoteIndexes.indexOf(i) !== -1) {
          this.makeKey(
              index, x, y, this.config.whiteWideNoteWidth,
              this.config.whiteNoteHeight, this.config.blackNoteWidth, 'white',
              '#141E30');
          x += this.config.whiteWideNoteWidth;
        } else if (whiteNarrowNoteIndexes.indexOf(i) !== -1) {
          this.makeKey(
              index, x, y, this.config.whiteNarrowNoteWidth,
              this.config.whiteNoteHeight, this.config.blackNoteWidth, 'white',
              '#141E30');
          x += this.config.whiteNarrowNoteWidth;
        }
        index++;
      }
    }

    if (this.octaves > 6) {
      // And an extra C at the end (if we're using all the octaves);
      this.makeKey(
          index, x, y, this.config.whiteHighestCWidth,
          this.config.whiteNoteHeight, this.config.blackNoteWidth, 'white',
          '#141E30');

      // Now draw all the black notes, so that they sit on top.
      // Pianos start on an A, if octave > 6, draw the leftmost A and A#
      this.makeKey(
          1, this.config.whiteNarrowNoteWidth - halfABlackNote, y,
          this.config.blackNoteWidth, this.config.blackNoteHeight,
          this.config.blackNoteWidth, 'black');
      index = 4;
      x = 2 * this.config.whiteNarrowNoteWidth;
    } else {
      // Starting 3 semitones up on small screens (on a C), and a whole octave
      // up.
      index = 4 + (this.lowestC - 1) * CONSTANTS.NOTES_PER_OCTAVE;
      x = 0;
    }

    // Draw the black notes.
    for (let o = 0; o < this.octaves; o++) {
      this.drawBlackNotesPerOctave(x, y, index, halfABlackNote);
      x += CONSTANTS.NOTES_PER_OCTAVE * this.config.blackNoteWidth;
      index += CONSTANTS.NOTES_PER_OCTAVE;
    }
  }

  /**
   * Set number of bars for cycle for when notes render in cycle
   * @param {number} measureNum
   */
  setCycle(measureNum) {
    // duration: bars
    let offset = this.config.whiteNoteHeight;
    let beatHeight = (this.height - offset) / measureNum / 4;
    // draw a vertical line for each beat
    for (let i = 0; i < measureNum * 4; i++) {
      let line = document.createElementNS(this.svgNS, 'line');
      line.setAttribute('x1', `0px`);
      line.setAttribute('y1', `${offset + i * beatHeight}px`);
      line.setAttribute('x2', `${this.width}px`);
      line.setAttribute('y2', `${offset + i * beatHeight}px`);
      line.setAttribute('stroke', '#5C5C5C');
      line.setAttribute('stroke-width', '1px');
      this.svg.appendChild(line);
    }
  }
}

/*************************
 * Floaty notes
 ************************/
class BaseFloatyNotes {
  /** Constructor for moving notes base class */
  constructor(canvas, width, height, x, y) {
    this.notes = [];  // the notes floating on the screen.
    this.canvas = canvas;
    this.context = this.canvas.getContext('2d');
    this.x = x;
    this.y = y;
    this.canvas.style.top = `${y}px`;
    this.canvas.style.left = `${x}px`;

    this.contextWidth = width;
    this.contextHeight = height;

    this.heldButtonToVisualData = new Map();

    this.started = false;
  }

  /** Clear all notes */
  clear() {
    this.notes = [];
  }

  /** Callback for window resizing */
  resize() {
    throw new Error('Implement Required');
  }

  /** Start animation */
  start() {
    this.started = true;
  }

  /** Stop animation */
  stop() {
    this.started = false;
  }

  /**
   * Set number of cycles for when rendering notes in cycle
   * @param {number} measureNum
   */
  setCycle(measureNum) {
    // by default, do nothing.
  }

  /**
   * Add note to grid
   * @param {string} color
   * @param {number} y
   * @param {number} height
   */
  addNote(color, y, height) {
    throw new Error('Implement Required');
  }

  /**
   * Remove note from grid
   * @param {Object!} noteToPaint
   */
  stopNote(noteToPaint) {
    throw new Error('Implement Required');
  }

  /** Draw all notes in grid, called every visual frame */
  draw() {
    throw new Error('Implement Required');
  }
}

class HorizontalFloatyNotes extends BaseFloatyNotes {
  /** Construct horizontal scrolling notes */
  constructor(canvas, width, height, x, y) {
    super(canvas, width, height, x, y);
  }

  /**
   * Resize window
   * @param {number} whiteNoteWidth
   */
  resize(whiteNoteWidth) {
    this.canvas.height = this.contextHeight;
    this.canvas.width = this.contextWidth = this.contextWidth - whiteNoteWidth;
    this.canvas.style.left = `${this.x + whiteNoteWidth}px`;
  }

  /**
   * Add note to grid
   * @param {string} color
   * @param {number} y
   * @param {number} height
   * @return {Object!}
   */
  addNote(color, y, height) {
    const noteToPaint = {
      x: 0,
      y: parseFloat(y),
      width: 0,
      height: parseFloat(height),
      color: CONSTANTS.COLORS[color],
      on: true
    };
    console.log('adding note', noteToPaint);
    // console.log(noteToPaint.color);
    this.notes.push(noteToPaint);
    // console.log(this.notes);
    return noteToPaint;
  }

  /**
   * Remove note from grid
   * @param {Object!} noteToPaint
   */
  stopNote(noteToPaint) {
    this.notes.forEach(note => {
      if (note.y === noteToPaint.y) {
        note.on = false;
      }
    });
  }

  /**
   * Draw all notes in the grid, scroll them down, remove ones offscreen.
   */
  draw() {
    const dx = 3;
    this.context.clearRect(0, 0, this.contextWidth, this.contextHeight);

    // Remove all the notes that will be off the page;
    this.notes = this.notes.filter(
        (note) => note.on || note.x < (this.contextWidth - 100));

    // Advance all the notes.
    for (let i = 0; i < this.notes.length; i++) {
      const note = this.notes[i];

      // If the note is still on, then its height goes up but it
      // doesn't start sliding down yet.
      if (note.on) {
        note.width += dx;
      } else {
        note.x += dx;
      }

      // the alpha of each note is smaller as it goes off the page.
      this.context.globalAlpha =
          1 - note.x / this.contextWidth;  // fade out as it goes off the page.
      this.context.fillStyle = note.color;
      this.context.fillRect(note.x, note.y, note.width, note.height);
    }
  }
}

class VerticalFloatyNotes extends BaseFloatyNotes {
  /** Construct vertical scrolling notes */
  constructor(canvas, width, height, x, y) {
    super(canvas, width, height, x, y);
  }

  /**
   * Resize window
   * @param {number} whiteNoteHeight
   */
  resize(whiteNoteHeight) {
    this.canvas.width = this.contextWidth;
    // Set the canvas to the right position that is the edge of the pianoroll
    // keyboard
    this.canvas.height = this.contextHeight =
        this.contextHeight - whiteNoteHeight;
    this.canvas.style.top = `${this.y + whiteNoteHeight}px`;
  }

  /**
   * Add note to grid
   * @param {string} color
   * @param {number} x
   * @param {number} width
   * @return {Object!}
   */
  addNote(color, x, width) {
    const noteToPaint = {
      x: parseFloat(x),
      y: 0,
      width: parseFloat(width),
      height: 0,
      color: CONSTANTS.COLORS[color],
      on: true
    };
    this.notes.push(noteToPaint);
    return noteToPaint;
  }

  /**
   * Remove note from grid
   * @param {Object!} noteToPaint
   */
  stopNote(noteToPaint) {
    this.notes.forEach(note => {
      if (note.x === noteToPaint.x) {
        note.on = false;
      }
    });
  }

  /**
   * Draw all notes in the grid, scroll them down, remove ones offscreen.
   */
  draw() {
    const dy = 3;  // the speed of the moving of note, 3px per frame
    this.context.clearRect(0, 0, this.contextWidth, this.contextHeight);

    // Remove all the notes that will be off the page;
    this.notes = this.notes.filter(
        (note) => note.on || note.y < (this.contextHeight - 100));

    // Advance all the notes.
    for (let i = 0; i < this.notes.length; i++) {
      const note = this.notes[i];

      // If the note is still on, then its height goes up but it
      // doesn't start sliding down yet.
      if (note.on) {
        note.height += dy;
      } else {
        note.y += dy;
      }

      this.context.globalAlpha = 1 - note.y / this.contextHeight;
      this.context.fillStyle = note.color;
      this.context.fillRect(note.x, note.y, note.width, note.height);
    }
  }
}

class VerticalAnticipationNotes extends BaseFloatyNotes {
  /** Construct vertical anticipation notes */
  constructor(canvas, width, height, x, y, pianoAtBottom) {
    super(canvas, width, height, x, y);
    this.visibleFrames = 4;
    this.pianoAtBottom = pianoAtBottom;
  }

  /**
   * Resize window
   * @param {number} whiteNoteHeight
   */
  resize(whiteNoteHeight) {
    this.canvas.width = this.contextWidth = window.innerWidth;
    // Set the canvas to the right position that is the edge of the pianoroll
    // keyboard
    this.canvas.height = this.contextHeight =
        window.innerHeight - whiteNoteHeight;
    if (this.pianoAtBottom) {
      this.canvas.style.top = `${this.y}px`;
    } else {
      this.canvas.style.top = `${this.y + whiteNoteHeight}px`;
    }
  }

  /**
   * Set number of frames notes are visible for
   * @param {number} visibleFrames
   */
  setVisibleFrames(visibleFrames) {
    this.visibleFrames = visibleFrames;
  }

  /**
   * Schedule a note onset
   * @param {number} noteNumber
   * @param {number} startFrame
   * @param {string} color
   * @param {number} alpha
   * @param {string} symbol
   * @param {number} x
   * @param {number} width
   */
  scheduleNoteOn(noteNumber, startFrame, color, alpha, symbol, x, width) {
    this.notes.push({
      noteNumber,
      symbol,
      startFrame,
      x,
      width,
      color: CONSTANTS.COLORS[color],
      alpha,
      stopFrame: undefined
    });
  }

  /**
   * Schedule a note release, find correct one in case multiple scheduled
   * @param {number} noteNumber
   * @param {number} stopFrame
   */
  scheduleNoteOff(noteNumber, stopFrame) {
    // Find corresponding note that is scheduled closest to stopFrame
    let closestNote;
    for (const note of this.notes) {
      if (note.noteNumber === noteNumber && note.startFrame < stopFrame &&
          (!closestNote || note.startFrame > closestNote.startFrame)) {
        closestNote = note;
      }
    }
    if (closestNote) {
      closestNote.stopFrame = stopFrame;
    }
  }

  /**
   * Clear all notes after target frame and clear stop of notes that stop after
   * target frame
   * @param {number} targetFrame
   */
  clearScheduledNotes(targetFrame) {
    // Clear all notes that start after target frame
    this.notes = this.notes.filter(note => note.startFrame < targetFrame);
    // Unset stopFrame on currently playing notes
    for (const note of this.notes) {
      if (note.stopFrame && note.stopFrame >= targetFrame) {
        note.stopFrame = undefined;
      }
    }
  }

  /** Render all notes in grid and remove ones offscreen */
  draw() {
    this.context.clearRect(0, 0, this.contextWidth, this.contextHeight);

    const curFrame = getTransportFrame();

    // Remove all the notes that will be off the page
    this.notes =
        this.notes.filter(note => !note.stopFrame || note.stopFrame > curFrame);

    // Render all the notes
    for (let i = 0; i < this.notes.length; i++) {
      const note = this.notes[i];

      const yStartRel = note.stopFrame ?
          1 - (note.stopFrame - curFrame) / this.visibleFrames :
          0;
      const yEndRel = 1 - (note.startFrame - curFrame) / this.visibleFrames;
      const gapSize = 10;  // Helps distinguish between repeated notes
      const y = this.contextHeight * yStartRel + gapSize;
      const height = this.contextHeight * (yEndRel - yStartRel) - gapSize;

      this.context.globalAlpha = note.alpha;
      this.context.fillStyle = note.color;
      this.context.fillRect(note.x, y, note.width, height);
      if (note.symbol) {
        this.context.fillStyle = 'white';
        this.context.font = '36px Arial';
        this.context.fillText(
            note.symbol, note.x - 40,
            Math.min(y + height, this.contextHeight - 10));
      }
    }
  }
}

class HorizontalSequencerNotes extends BaseFloatyNotes {
  /** Construct horizontal cycle notes */
  constructor(canvas, width, height, x, y) {
    super(canvas, width, height, x, y);
    this.position = 0;
    this.startTime = 0;
    this.cycleSeconds = 0;
    this.refreshRate = CONSTANTS.REFRESH_RATE;
  }

  /** Clear all old notes */
  clear() {
    this.notes = this.notes.filter((note) => note.on);
    this.notes.forEach((note) => {
      note.x = 0;
      note.width = 0;
    });
    this.startTime = Date.now();
  }

  /** Start cycle */
  start() {
    this.started = true;
    this.startTime = Date.now();
  }

  /** Stop cycle */
  stop() {
    this.started = false;
  }

  /**
   * Set cycle beats
   * @param {number} measureNum
   */
  setCycle(measureNum) {
    this.cycleSeconds = ((60 / Tone.Transport.bpm.value * 4) * measureNum);
  }

  /**
   * Callback after window resize
   * @param {number} whiteNoteWidth
   */
  resize(whiteNoteWidth) {
    this.canvas.height = this.contextHeight;
    this.canvas.width = this.contextWidth = this.contextWidth - whiteNoteWidth;
    this.canvas.style.left = `${this.x + whiteNoteWidth}px`;
  }

  /**
   * Add note to grid
   * @param {string} color
   * @param {number} y
   * @param {number} height
   * @return {Object!}
   */
  addNote(color, y, height) {
    const noteToPaint = {
      x: this.position,
      y: parseFloat(y),
      width: 0,
      height: parseFloat(height),
      color: CONSTANTS.COLORS[color],
      on: true
    };
    console.log('adding note', noteToPaint);
    this.notes.push(noteToPaint);
    return noteToPaint;
  }

  /**
   * Stop playing note
   * @param {Object!} noteToPaint
   */
  stopNote(noteToPaint) {
    this.notes.forEach(note => {
      if (note.y === noteToPaint.y) {
        note.on = false;
      }
    });
  }

  /** Draw notes on grid and remove old ones */
  draw() {
    if (this.started) {
      this.position = (Date.now() - this.startTime) / 1000 % this.cycleSeconds *
          this.contextWidth / this.cycleSeconds;

      // clear all the lines and rectangles
      this.context.clearRect(0, 0, this.contextWidth, this.contextHeight);

      this.context.beginPath();               // Start a new path
      this.context.moveTo(this.position, 0);  // Move the pen to
      this.context.lineTo(this.position, this.canvas.height);  // Draw a line to
      this.context.strokeStyle = 'red';
      this.context.stroke();  // Render the path

      // Remove all the notes that will be off the page;
      this.notes =
          this.notes.filter((note) => note.on || note.x < (this.contextWidth));

      // Advance all the notes.
      for (let i = 0; i < this.notes.length; i++) {
        const note = this.notes[i];

        // If the note is still on, then its height goes up but it
        // doesn't start sliding down yet.
        if (note.on) {
          note.width = this.position - note.x;
        }

        this.context.fillStyle = note.color;
        this.context.fillRect(note.x, note.y, note.width, note.height);
      }
    }
  }
}

class VerticalSequencerNotes extends BaseFloatyNotes {
  /** Construct vertical sequencer notes */
  constructor(canvas, width, height, x, y) {
    super(canvas, width, height, x, y);
    this.position = 0;
    this.startTime = 0;
    this.cycleSeconds = 0;
    this.refreshRate = CONSTANTS.REFRESH_RATE;
  }

  /** Clear all old notes */
  clear() {
    this.notes = this.notes.filter((note) => note.on);
    this.notes.forEach((note) => {
      note.y = 0;
      note.height = 0;
    });
    // this.notes = [];
    this.startTime = Date.now();
  }

  /** Start cycle */
  start() {
    this.started = true;
    this.startTime = Date.now();
  }

  /** Stop cycle */
  stop() {
    this.started = false;
  }

  /**
   * Set cycle beats
   * @param {number} measureNum
   */
  setCycle(measureNum) {
    this.cycleSeconds = ((60 / Tone.Transport.bpm.value * 4) * measureNum);
  }

  /**
   * Callback after window resize
   * @param {number} whiteNoteHeight
   */
  resize(whiteNoteHeight) {
    this.canvas.width = this.contextWidth;
    // Set the canvas to the right position that is the edge of the pianoroll
    // keyboard
    this.canvas.height = this.contextHeight =
        this.contextHeight - whiteNoteHeight;
    this.canvas.style.top = `${this.y + whiteNoteHeight}px`;
  }

  /**
   * Add note to grid
   * @param {string} color
   * @param {number} x
   * @param {number} width
   * @return {Object!}
   */
  addNote(color, x, width) {
    const noteToPaint = {
      x: parseFloat(x),
      y: this.position,
      width: parseFloat(width),
      height: 0,
      color: CONSTANTS.COLORS[color],
      on: true
    };
    this.notes.push(noteToPaint);
    return noteToPaint;
  }

  /**
   * Stop playing note
   * @param {Object!} noteToPaint
   */
  stopNote(noteToPaint) {
    this.notes.forEach(note => {
      if (note.x === noteToPaint.x) {
        note.on = false;
      }
    });
  }

  /** Draw notes on grid and remove old ones */
  draw() {
    if (this.started) {
      this.position = (Date.now() - this.startTime) / 1000 % this.cycleSeconds *
          this.contextHeight / this.cycleSeconds;

      this.context.clearRect(0, 0, this.contextWidth, this.contextHeight);

      this.context.beginPath();               // Start a new path
      this.context.moveTo(0, this.position);  // Move the pen to
      this.context.lineTo(this.canvas.width, this.position);  // Draw a line to
      this.context.strokeStyle = 'red';
      this.context.stroke();  // Render the path

      // Remove all the notes that will be off the page;
      this.notes =
          this.notes.filter((note) => note.on || note.y < (this.contextHeight));

      // Advance all the notes.
      for (let i = 0; i < this.notes.length; i++) {
        const note = this.notes[i];

        // If the note is still on, then its height goes up but it
        // doesn't start sliding down yet.
        if (note.on) {
          note.height = this.position - note.y;
        }

        // this.context.globalAlpha = 1 - note.y / this.contextHeight;
        this.context.fillStyle = note.color;
        this.context.fillRect(note.x, note.y, note.width, note.height);
      }
    }
  }
}

class HorizontalSequencerOverlapNotes extends BaseFloatyNotes {
  /** Construct overlapping notes */
  constructor(canvas, width, height, x, y) {
    super(canvas, width, height, x, y);
    this.position = 0;
    this.started = false;
    this.dx = 1;
    this.refreshRate = CONSTANTS.REFRESH_RATE;
    this.preregisteredNotes = [];
    this.drawPreRegisteredNotes = 'off';
    this.preregisteredNotesAlpha = 1.0;
    this.playNotesAlpha = 1.0;
  }

  /** Remove old notes */
  clear() {
    this.notes = this.notes.filter((note) => note.on);
    this.notes.forEach((note) => {
      note.x = 0;
      note.width = 0;
    });
    // this.notes = [];
    this.startTime = Date.now();
  }

  /** Start cycle */
  start() {
    this.started = true;
    this.startTime = Date.now();
  }

  /** Stop cycle */
  stop() {
    this.started = false;
  }

  /**
   * Set cycle duration in beats
   * @param {number} duration
   */
  setCycle(duration) {
    this.cycleSeconds = ((60 / Tone.Transport.bpm.value * 4) * duration);
  }

  /**
   * Callback after window resize
   * @param {number} whiteNoteWidth
   */
  resize(whiteNoteWidth) {
    this.canvas.height = this.contextHeight;
    this.canvas.width = this.contextWidth = this.contextWidth - whiteNoteWidth;
    this.canvas.style.left = `${this.x + whiteNoteWidth}px`;
  }

  /**
   * Add note to grid
   * @param {string} color
   * @param {number} y
   * @param {number} height
   * @return {Object!}
   */
  addNote(color, y, height) {
    const noteToPaint = {
      x: this.position,
      y: parseFloat(y),
      width: 0,
      height: parseFloat(height),
      color: CONSTANTS.COLORS[color],
      on: true
    };
    console.log('adding note', noteToPaint);
    this.notes.push(noteToPaint);
    return noteToPaint;
  }

  /**
   * Stop playing note
   * @param {Object!} noteToPaint
   */
  stopNote(noteToPaint) {
    this.notes.forEach(note => {
      if (note.y === noteToPaint.y) {
        note.on = false;
      }
    });
  }

  /** Draw notes on grid, remove old ones, and advance */
  draw() {
    if (this.started) {
      this.position = (Date.now() - this.startTime) / 1000 % this.cycleSeconds *
          this.contextWidth / this.cycleSeconds;

      // clear all the lines and rectangles
      this.context.clearRect(0, 0, this.contextWidth, this.contextHeight);

      this.context.globalAlpha = 1.0;
      this.context.beginPath();               // Start a new path
      this.context.moveTo(this.position, 0);  // Move the pen to
      this.context.lineTo(this.position, this.canvas.height);  // Draw a line to
      this.context.strokeStyle = 'red';
      this.context.stroke();  // Render the path

      // Remove all the notes that will be off the page;
      this.notes =
          this.notes.filter((note) => note.on || note.x < (this.contextWidth));

      // Advance all the notes.
      for (let i = 0; i < this.notes.length; i++) {
        const note = this.notes[i];

        // If the note is still on, then its height goes up but it
        // doesn't start sliding down yet.
        if (note.on) {
          note.width = this.position - note.x;
        }

        this.context.fillStyle = note.color;
        this.context.globalAlpha = this.playNotesAlpha;
        this.context.fillRect(note.x, note.y, note.width, note.height);
      }

      // Advance all the notes.
      if (this.drawPreRegisteredNotes === 'appear' ||
          this.drawPreRegisteredNotes === 'on') {
        for (let i = 0; i < this.preregisteredNotes.length; i++) {
          const note = this.preregisteredNotes[i];
          let displayWidth;

          // If the note is still on, then its height goes up but it
          // doesn't start sliding down yet.
          if (note.x < this.position || this.drawPreRegisteredNotes === 'on') {
            if (note.x + note.width > this.position &&
                this.drawPreRegisteredNotes === 'appear') {
              displayWidth = Math.min(this.position - note.x, note.width);
            } else {
              displayWidth = note.width;
            }
            this.context.fillStyle = note.color;
            this.context.globalAlpha = this.preregisteredNotesAlpha;
            this.context.fillRect(note.x, note.y, displayWidth, note.height);
          }
        }
      }
    }
  }
}


class NoteVisual {
  /** Construct manager class for grid notes and piano */
  constructor(
      div, animationType = 'waterfall', orientation = 'vertical',
      numOctaves = -1, lowestC = 1, width = -1, height = -1, x = 0, y = 0,
      pianoAtBottom = true) {
    // TODO: add a flag of piano layout so that we don't need to check
    // numOctaves
    this.animationType = animationType;
    this.orientation = orientation;
    this.numOctaves = numOctaves;

    // let divSelected = document.querySelector(div);
    let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('id', 'svg');
    let canvas = document.createElement('canvas');
    canvas.setAttribute('id', 'canvas');
    div.append(svg);
    div.append(canvas);

    if (this.numOctaves === 0 || this.numOctaves < -1) {
      throw new Error('numOctaves must be -1 or greater than 0');
    }
    if (width === -1) {
      width = window.innerWidth;
    }
    if (height === -1) {
      height = window.innerHeight;
    }
    this.lowestC = lowestC;
    if (this.numOctaves + this.lowestC > 8) {
      throw new Error(
          'Please lower the lowestC. The highest note cannot exceed C8.');
    }
    if (this.orientation === 'horizontal') {
      this.piano = new HorizontalPiano(svg, width, height, x, y, lowestC);
      if (this.animationType === 'sequencer') {
        this.painter =
            new HorizontalSequencerNotes(canvas, width, height, x, y);
      } else if (this.animationType === 'waterfall') {
        this.painter = new HorizontalFloatyNotes(canvas, width, height, x, y);
      } else if (this.animationType === 'sequencer-overlap') {
        this.painter =
            new HorizontalSequencerOverlapNotes(canvas, width, height, x, y);
      }
    } else if (this.orientation === 'vertical') {
      this.piano =
          new VerticalPiano(svg, width, height, x, y, lowestC, pianoAtBottom);
      if (this.animationType === 'sequencer') {
        this.painter = new VerticalSequencerNotes(canvas, width, height, x, y);
      } else if (this.animationType === 'waterfall') {
        this.painter = new VerticalFloatyNotes(canvas, width, height, x, y);
      } else if (this.animationType === 'anticipation') {
        this.painter = new VerticalAnticipationNotes(
            canvas, width, height, x, y, pianoAtBottom);
      }
    }
    this.onWindowResize();
  }

  /**
   * Set octave computer keyboard starts at
   * @param {number} computerKeyOctave
   */
  setComputerKeyOctave(computerKeyOctave) {
    this.piano.setComputerKeyOctave(computerKeyOctave);
  }

  /**
   * Set number of frames visible in grid for anticipation notes
   * @param {number} visibleFrames
   */
  setVisibleFrames(visibleFrames) {
    if (this.animationType !== 'anticipation') {
      throw new Error('Only supported for anticipation');
    }
    this.painter.setVisibleFrames(visibleFrames);
  }

  /**
   * Highlight note on piano and add to grid if not anticipation
   * @param {number} noteNumber
   * @param {string=} color
   */
  noteOn(noteNumber, color = 'orange') {
    const rect = this.piano.highlightNote(
        noteNumber - CONSTANTS.LOWEST_PIANO_KEY_MIDI_NOTE, color);
    let noteToPaint;
    if (this.animationType !== 'anticipation') {
      if (this.orientation === 'horizontal') {
        noteToPaint = this.painter.addNote(
            color, rect.getAttribute('y_position'),
            rect.getAttribute('noteSize'));

      } else if (this.orientation === 'vertical') {
        noteToPaint = this.painter.addNote(
            color, rect.getAttribute('x_position'),
            rect.getAttribute('noteSize'));
      }
    }

    this.painter.heldButtonToVisualData.set(
        noteNumber, {rect: rect, note: noteNumber, noteToPaint: noteToPaint});
  }

  /**
   * Stop playing a note on the piano and for non-anticipation tell scroller
   * that note should stop
   * @param {number} noteNumber
   */
  noteOff(noteNumber) {
    const note = this.painter.heldButtonToVisualData.get(noteNumber);
    if (note) {
      // Don't see it.
      this.piano.clearNote(note.rect);

      // Stop holding it down.
      if (this.animationType !== 'anticipation') {
        this.painter.stopNote(note.noteToPaint);
      }
    }
    this.painter.heldButtonToVisualData.delete(noteNumber);
  }

  /** Stop playing all notes that are currently held */
  stopAllNotes() {
    for (const noteNumber of Array.from(
             this.painter.heldButtonToVisualData.keys())) {
      this.noteOff(noteNumber);
    }
  }

  /**
   * For anticipation, schedule a note to be played at given frame
   * @param {number} noteNumber
   * @param {number} frame
   * @param {string} color
   * @param {number} alpha
   * @param {string=} symbol
   */
  scheduleNoteOn(noteNumber, frame, color, alpha, symbol = '') {
    if (this.animationType !== 'anticipation') {
      throw new Error('Only supported for anticipation');
    }
    const rect = this.piano.findNoteRect(
        noteNumber - CONSTANTS.LOWEST_PIANO_KEY_MIDI_NOTE);
    const pos = this.orientation === 'horizontal' ?
        rect.getAttribute('y_position') :
        rect.getAttribute('x_position');
    this.painter.scheduleNoteOn(
        noteNumber, frame, color, alpha, symbol, pos,
        rect.getAttribute('noteSize'));
  }

  /**
   * For anticipation, schedule a note to be released at given frame
   * @param {number} noteNumber
   * @param {number} frame
   */
  scheduleNoteOff(noteNumber, frame) {
    if (this.animationType !== 'anticipation') {
      throw new Error('Only supported for anticipation');
    }
    this.painter.scheduleNoteOff(noteNumber, frame);
  }

  /**
   * For anticipation, release all notes after frame
   * @param {number} frame
   */
  clearScheduledNotes(frame) {
    if (this.animationType !== 'anticipation') {
      throw new Error('Only supported for anticipation');
    }
    this.painter.clearScheduledNotes(frame);
  }

  /** Callback for window resizing */
  onWindowResize() {
    this.piano.octaves = this.numOctaves;
    if (this.orientation === 'horizontal') {
      if (this.numOctaves === -1) {
        this.piano.octaves = this.piano.height > 700 ? 7 : 3;
      }
      this.painter.resize(this.piano.config.whiteNoteWidth);
    } else if (this.orientation === 'vertical') {
      if (this.numOctaves === -1) {
        this.piano.octaves = this.piano.width > 700 ? 7 : 3;
      }
      this.painter.resize(this.piano.config.whiteNoteHeight);
    }
    const bonusNotes =
        this.piano.octaves > 6 ? 4 : 0;  // starts on an A, ends on a C.
    const additionalWhiteNotes = bonusNotes > 0 ? 3 : 0;
    const totalWhiteNotes =
        CONSTANTS.WHITE_NOTES_PER_OCTAVE * this.piano.octaves +
        additionalWhiteNotes;

    this.piano.resize(totalWhiteNotes);
    this.piano.draw();
    this.piano.drawBackgroundGrid();
  }

  /**
   * For cycle notes, set number of cycle bars
   * @param {number} cycle
   */
  setCycle(cycle) {
    // cycle: in number of bars
    this.painter.setCycle(cycle);
    this.piano.setCycle(cycle);
  }

  /** Start animating */
  start() {
    this.painter.start();  // start the painter animation
  }

  /** Stop animating */
  stop() {
    this.painter.stop();  // stop the painter animation
  }

  /** Clear notes */
  clear() {
    this.painter.clear();  // clear the painter animation
  }
}

class DrawLoop {
  /** Construct draw loop with visual fps */
  constructor(fps = 60) {
    // Limit the refresh rate to 60fps
    // https://stackoverflow.com/questions/19764018/controlling-fps-with-requestanimationframe
    this.drawFunction = [];
    this.fps = fps;
    this.fpsInterval = 1000 / fps;
    this.then = Date.now();
    this.startTime = this.then;
  }

  /** Add draw function to call every frame */
  addDrawFunction(drawFunction) {
    this.drawFunction.push(drawFunction);
  }

  /** Add draw function from visual painter */
  addDrawFunctionFromVisual(visual) {
    this.addDrawFunction(visual.painter.draw.bind(visual.painter));
  }

  /** Call draw functions every frame and recursively call this function */
  startDrawLoop() {
    // The refresh rate of the animation is determined by the refresh rate of
    // the monitor. So in different monitors, the call to
    // window.requestAnimationFrame will have different refresh rates. After the
    // next line, we will control the refresh rate to be constant.
    window.requestAnimationFrame(() => this.startDrawLoop());


    // calc elapsed time since last loop
    this.now = Date.now();
    let elapsed = this.now - this.then;

    // if enough time has elapsed, draw the next frame

    if (elapsed > this.fpsInterval) {
      // Get ready for next frame by setting then=now, but...
      // Also, adjust for fpsInterval not being multiple of 16.67
      this.then = this.now - (elapsed % this.fpsInterval);

      // draw stuff here
      for (let i = 0; i < this.drawFunction.length; i++) {
        this.drawFunction[i]();
      }
    }
  }
}
