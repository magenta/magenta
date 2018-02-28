/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import * as test from "tape";
import {tensorflow} from './proto';
import NoteSequence = tensorflow.magenta.NoteSequence;
import {Sequences} from './sequences';

const STEPS_PER_QUARTER = 4;

function createTestNS() {
  const ns = NoteSequence.create();

  ns.tempos.push(NoteSequence.Tempo.create({qpm: 60, time: 0}));

  ns.timeSignatures.push(NoteSequence.TimeSignature.create({
    time: 0,
    numerator: 4,
    denominator: 4,
  }));

  return ns;
}

function addTrackToSequence(ns: NoteSequence, instrument: number,
    notes: number[][]) {
  for (const noteParams of notes) {
    const note = new NoteSequence.Note({
      pitch: noteParams[0],
      velocity: noteParams[1],
      startTime: noteParams[2],
      endTime: noteParams[3]
    });
    ns.notes.push(note);
    if(ns.totalTime < note.endTime) {
        ns.totalTime = note.endTime;
    }
  }
}

function addChordsToSequence(ns: NoteSequence,
    chords: Array<Array<number|string>>) {
  for (const chordParams of chords) {
      const ta = NoteSequence.TextAnnotation.create({
        text: chordParams[0] as string,
        time: chordParams[1] as number,
        annotationType:
          NoteSequence.TextAnnotation.TextAnnotationType.CHORD_SYMBOL
      });
      ns.textAnnotations.push(ta);
  }
}

function addControlChangesToSequence(ns: NoteSequence, instrument:number,
    chords:number[][]) {
  for (const ccParams of chords) {
    const cc = NoteSequence.ControlChange.create({
      time: ccParams[0],
      controlNumber: ccParams[1],
      controlValue: ccParams[2],
      instrument
    });
    ns.controlChanges.push(cc);
  }
}

function addQuantizedStepsToSequence(ns: NoteSequence,
    quantizedSteps: number[][]) {
  quantizedSteps.forEach((qstep, i) => {
    const note = ns.notes[i];
    note.quantizedStartStep = qstep[0];
    note.quantizedEndStep = qstep[1];
    if(note.quantizedEndStep > ns.totalQuantizedSteps) {
      ns.totalQuantizedSteps = note.quantizedEndStep;
    }
  });
}

function addQuantizedChordStepsToSequence(ns:NoteSequence,
    quantizedSteps:number[]) {
  const chordAnnotations = ns.textAnnotations.filter(
    ta => ta.annotationType ===
    NoteSequence.TextAnnotation.TextAnnotationType.CHORD_SYMBOL);

  quantizedSteps.forEach((qstep, i) => {
    const ta = chordAnnotations[i];
    ta.quantizedStep = qstep;
  });
}

function addQuantizedControlStepsToSequence(ns:NoteSequence,
    quantizedSteps:number[]) {
  quantizedSteps.forEach((qstep, i) => {
      const cc = ns.controlChanges[i];
      cc.quantizedStep = qstep;
  });
}

test("Quantize NoteSequence", (t:test.Test) => {
    const ns = createTestNS();

    addTrackToSequence(
        ns, 0,
        [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
        [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
    addChordsToSequence(
        ns,
        [['B7', 0.22], ['Em9', 4.0]]);
    addControlChangesToSequence(
        ns, 0,
        [[2.0, 64, 127], [4.0, 64, 0]]);

    // Make a copy.
    const expectedQuantizedSequence = NoteSequence.decode(
        NoteSequence.encode(ns).finish());
    expectedQuantizedSequence.quantizationInfo =
        NoteSequence.QuantizationInfo.create({
          stepsPerQuarter: STEPS_PER_QUARTER
        });
    expectedQuantizedSequence.quantizationInfo.stepsPerQuarter =
        STEPS_PER_QUARTER;
    addQuantizedStepsToSequence(
        expectedQuantizedSequence,
        [[0, 40], [1, 2], [10, 14], [16, 17], [19, 20]]);
    addQuantizedChordStepsToSequence(
        expectedQuantizedSequence, [1, 16]);
    addQuantizedControlStepsToSequence(
        expectedQuantizedSequence, [8, 16]);

    const qns = Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

    t.deepEqual(NoteSequence.toObject(qns),
        NoteSequence.toObject(expectedQuantizedSequence));

    t.end();
});
