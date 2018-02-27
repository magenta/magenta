/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as dl from 'deeplearn';

const DEFAULT_DRUM_PITCH_CLASSES: number[][] = [
  // bass drum
  [36, 35],
  // snare drum
  [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],
  // closed hi-hat
  [42, 44, 54, 68, 69, 70, 71, 73, 78, 80],
  // open hi-hat
  [46, 67, 72, 74, 79, 81],
  // low tom
  [45, 29, 41, 61, 64, 84],
  // mid tom
  [48, 47, 60, 63, 77, 86, 87],
  // high tom
  [50, 30, 43, 62, 76, 83],
  // crash cymbal
  [49, 55, 57, 58],
  // ride cymbal
  [51, 52, 53, 59, 82]
];

class Note {
  pitch: number;
  startStep: number;  // inclusive
  endStep: number;  // exclusive

  constructor(pitch: number, startStep: number, endStep?: number) {
    this.pitch = pitch;
    this.startStep = startStep;
    this.endStep = (endStep) ? endStep : startStep + 1;
  }

  public toString(): string {
    return (
      '{p:' + this.pitch + ' s:' + this.startStep + ' e:' + this.endStep + '}');
  }
}

abstract class DataConverter {
  abstract numSteps: number;
  abstract toTensor(notes: Note[]): dl.Tensor2D;
  abstract toNoteSequence(tensor: dl.Tensor2D): Note[];
}

export class DrumsConverter extends DataConverter{
  numSteps: number;
  pitchClasses: number[][];
  pitchToClass: {[pitch: number] : number};

  constructor(
      numSteps: number, pitchClasses: number[][]=DEFAULT_DRUM_PITCH_CLASSES) {
    super();
    this.numSteps = numSteps;
    this.pitchClasses = pitchClasses;
    this.pitchToClass = {};
    for (let c = 0; c < pitchClasses.length; ++c) {  // class
      pitchClasses[c].forEach((p) => {this.pitchToClass[p] = c;});
    }
  }

  toTensor(notes: Note[]) {
    const drumRoll = dl.buffer([this.numSteps, this.pitchClasses.length + 1]);
    // Set final values to 1 and change to 0 later if the has gets a note.
    for (let i = 0; i < this.numSteps; ++i) {
      drumRoll.set(1, i, -1);
    }
    notes.forEach((note) => {
      drumRoll.set(1, note.startStep, this.pitchToClass[note.pitch]);
      drumRoll.set(0, note.startStep, -1);
    });
    return drumRoll.toTensor() as dl.Tensor2D;
  }

  toNoteSequence(oh: dl.Tensor2D) {
    const notes: Note[] = [];
    const labelsTensor = oh.argMax(1);
    const labels: Int32Array = labelsTensor.dataSync() as Int32Array;
    labelsTensor.dispose();
    for (let s = 0; s < labels.length; ++s) {  // step
      for (let p = 0; p < this.pitchClasses.length; p++) {  // pitch class
        if (labels[s] >> p & 1) {
          notes.push(new Note(this.pitchClasses[p][0], s));
        }
      }
    }
    return notes;
  }
}

class DrumRollConverter extends DrumsConverter {
  toNoteSequence(roll: dl.Tensor2D) {
    const notes: Note[] = [];
    for (let s = 0; s < roll.shape[0]; ++s) {  // step
      const rollSlice = roll.slice([s, 0], [1, roll.shape[1]]);
      const pitches = rollSlice.dataSync() as Uint8Array;
      rollSlice.dispose();
      for (let p = 0; p < pitches.length; ++p) {  // pitch class
        if (pitches[p]) {
          notes.push(new Note(this.pitchClasses[p][0], s, s + 1));
        }
      }
    }
    return notes;
  }
}

export class MelodyConverter extends DataConverter{
  numSteps: number;
  minPitch: number;  // inclusive
  maxPitch: number;  // inclusive
  depth: number;
  NOTE_OFF = 1;
  FIRST_PITCH = 2;

  constructor(numSteps: number, minPitch=21, maxPitch=108) {
    super();
    this.numSteps = numSteps;
    this.minPitch = minPitch;
    this.maxPitch = maxPitch;
    this.depth = maxPitch - minPitch + 3;
  }

  toTensor(notes: Note[]) {
    notes = notes.sort((n1, n2) => n1.startStep - n2.startStep);
    const mel = dl.buffer([this.numSteps]);
    let lastEnd = -1;
    notes.forEach(n => {
      if  (n.startStep < lastEnd) {
        throw new Error('NoteSequence is not monophonic.');
      }
      mel.set(n.pitch - this.minPitch + this.FIRST_PITCH, n.startStep);
      mel.set(this.NOTE_OFF, n.endStep);
      lastEnd = n.endStep;
    });
    return dl.oneHot(
        mel.toTensor() as dl.Tensor1D, this.depth) as dl.Tensor2D;
  }

  toNoteSequence(oh: dl.Tensor2D) {
    const notes: Note[] = [];
    const labelsTensor = oh.argMax(1);
    const labels: Int32Array = labelsTensor.dataSync() as Int32Array;
    labelsTensor.dispose();
    let currNote: Note = null;
    for (let s = 0; s < labels.length; ++s) {  // step
      const label = labels[s];
      switch (label) {
        case 0:
          break;
        case 1:
          if (currNote) {
            currNote.endStep = s;
            notes.push(currNote);
            currNote = null;
          }
          break;
        default:
          if (currNote) {
            currNote.endStep = s;
            notes.push(currNote);
          }
          currNote = new Note(label - this.FIRST_PITCH + this.minPitch, s);
      }
    }
    if (currNote) {
      currNote.endStep = labels.length;
      notes.push(currNote);
    }
    return notes;
  }
}

function intsToBits(ints: number[], depth: number) {
  const bits: number[][] = [];
  for (let i = 0; i < ints.length; i++) {
    const b: number[] = [];
    for (let d = 0; d < depth; d++) {
      b.push(ints[i] >> d & 1);
    }
    if (ints[i] === 0) {
      b[depth - 1] = 1;
    }
    bits.push(b);
  }
  return bits;
}

function bitsToInts(bits: Uint8Array[]) {
  const ints: number[] = [];
  for (let i = 0; i < bits.length; i++) {
    let b = 0;
    for (let d = 0; d < bits[i].length; d++) {
      b += (bits[i][d] << d);
    }
    ints.push(b);
  }
  return ints;
}

function intsToOneHot(ints: number[], depth: number) {
  const oneHot: number[][] = [];
  for (let i = 0; i < ints.length; i++) {
    const oh: number[] = [];
    for (let d = 0; d < depth; d++) {
      oh.push(d === ints[i] ? 1 : 0);
    }
    oneHot.push(oh);
  }
  return oneHot;
}

export {
  Note,
  DataConverter,
  DrumRollConverter,
  bitsToInts,
  intsToBits,
  intsToOneHot,
};
