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
import {tensorflow} from '@magenta/protobuf';
import NoteSequence = tensorflow.magenta.NoteSequence;
import * as sequences from './sequences';
import Sequences = sequences.Sequences;

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

test("Quantize NoteSequence, Time Signature Change", (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
      [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
  ns.timeSignatures.length = 0;

  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Single time signature.
  ns.timeSignatures.push(NoteSequence.TimeSignature.create({
    numerator: 4, denominator: 4, time: 0}));
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Multiple time signatures with no change.
  ns.timeSignatures.push(NoteSequence.TimeSignature.create({
    numerator: 4, denominator: 4, time: 1}));
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Time signature change.
  ns.timeSignatures.push(NoteSequence.TimeSignature.create({
    numerator: 2, denominator: 4, time: 2}));
  t.throws(
    () => Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER),
    sequences.MultipleTimeSignatureException);

  t.end();
});

test("Quantize NoteSequence, Implicit Time Signature Change", (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
      [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
  ns.timeSignatures.length = 0;

  // No time signature.
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Implicit time signature change.
  ns.timeSignatures.push(NoteSequence.TimeSignature.create({
    numerator: 2, denominator: 4, time: 2}));
  t.throws(
    () => Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER),
    sequences.MultipleTimeSignatureException);

  t.end();
});

test("Quantize NoteSequence, No Implicit Time Signature Change, Out Of Order",
    (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
      [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
  ns.timeSignatures.length = 0;

  // No time signature.
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // No implicit time signature change, but time signatures are added out of
  // order.
  ns.timeSignatures.push(NoteSequence.TimeSignature.create({
    numerator: 2, denominator: 4, time: 2}));
  ns.timeSignatures.push(NoteSequence.TimeSignature.create({
    numerator: 2, denominator: 4, time: 0}));
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);
  t.pass();

  t.end();
});

test("StepsPerQuarterToStepsPerSecond", (t:test.Test) => {
  t.equal(Sequences.stepsPerQuarterToStepsPerSecond(4, 60.0), 4.0);

  t.end();
});

test("QuantizeToStep", (t:test.Test) => {
  t.equal(Sequences.quantizeToStep(8.0001, 4), 32);
  t.equal(Sequences.quantizeToStep(8.4999, 4), 34);
  t.equal(Sequences.quantizeToStep(8.4999, 4, 1.0), 33);
  t.end();
});

test("Quantize NoteSequence, Tempo Change", (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
      [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
  ns.tempos.length = 0;

  // No tempos.
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Single tempo.
  ns.tempos.push(NoteSequence.Tempo.create({qpm: 60, time: 0}));
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Multiple tempos with no change.
  ns.tempos.push(NoteSequence.Tempo.create({qpm: 60, time: 1}));
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Tempo change.
  ns.tempos.push(NoteSequence.Tempo.create({qpm: 120, time: 2}));
  t.throws(
    () => Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER),
    sequences.MultipleTempoException);

  t.end();
});

test("Quantize NoteSequence, Implicit Tempo Change", (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
      [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
  ns.tempos.length = 0;

  // No tempos.
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // Implicit tempo change.
  ns.tempos.push(NoteSequence.Tempo.create({qpm: 60, time: 2}));
  t.throws(
    () => Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER),
    sequences.MultipleTempoException);

  t.end();
});

test("Quantize NoteSequence, No Implicit Tempo Change, Out of Order",
    (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
      [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
  ns.tempos.length = 0;

  // No tempos.
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

  // No implicit tempo change, but tempos are added out of order.
  ns.tempos.push(NoteSequence.Tempo.create({qpm: 60, time: 2}));
  ns.tempos.push(NoteSequence.Tempo.create({qpm: 60, time: 0}));
  Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);
  t.pass();

  t.end();
});

test("Quantize NoteSequence, Rounding", (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 1,
      [[12, 100, 0.01, 0.24], [11, 100, 0.22, 0.55], [40, 100, 0.50, 0.75],
       [41, 100, 0.689, 1.18], [44, 100, 1.19, 1.69], [55, 100, 4.0, 4.01]]);

  // Make a copy.
  const expectedQuantizedSequence = NoteSequence.decode(
      NoteSequence.encode(ns).finish());
  expectedQuantizedSequence.quantizationInfo =
      NoteSequence.QuantizationInfo.create({
        stepsPerQuarter: STEPS_PER_QUARTER
      });

  addQuantizedStepsToSequence(
      expectedQuantizedSequence,
      [[0, 1], [1, 2], [2, 3], [3, 5], [5, 7], [16, 17]]);

  const quantizedSequence = Sequences.quantizeNoteSequence(
    ns, STEPS_PER_QUARTER);

  t.deepEqual(NoteSequence.toObject(quantizedSequence),
      NoteSequence.toObject(expectedQuantizedSequence));

  t.end();
});

test("Quantize NoteSequence, MultiTrack", (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 1.0, 4.0], [19, 100, 0.95, 3.0]]);
  addTrackToSequence(
      ns, 3,
      [[12, 100, 1.0, 4.0], [19, 100, 2.0, 5.0]]);
  addTrackToSequence(
      ns, 7,
      [[12, 100, 1.0, 5.0], [19, 100, 2.0, 4.0], [24, 100, 3.0, 3.5]]);

  // Make a copy.
  const expectedQuantizedSequence = NoteSequence.decode(
      NoteSequence.encode(ns).finish());
  expectedQuantizedSequence.quantizationInfo =
      NoteSequence.QuantizationInfo.create({
        stepsPerQuarter: STEPS_PER_QUARTER
      });

  addQuantizedStepsToSequence(
      expectedQuantizedSequence,
      [[4, 16], [4, 12], [4, 16], [8, 20], [4, 20], [8, 16], [12, 14]]);

  const quantizedSequence = Sequences.quantizeNoteSequence(
    ns, STEPS_PER_QUARTER);

  t.deepEqual(NoteSequence.toObject(quantizedSequence),
      NoteSequence.toObject(expectedQuantizedSequence));

  t.end();
});

test("Assert isQuantizedNoteSequence", (t:test.Test) => {
  const ns = createTestNS();

  addTrackToSequence(
      ns, 0,
      [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
      [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);


  t.throws(
    () => Sequences.assertIsQuantizedSequence(ns),
    sequences.QuantizationStatusException);

  const qns = Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);
  Sequences.assertIsQuantizedSequence(qns);

  t.end();
});
