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

import * as magenta from '@magenta/core';
import NoteSequence = magenta.NoteSequence;
import { MelodyRnn } from '../src/index';

async function runMelodyRnn() {
  const melodyRnn = new MelodyRnn();
  await melodyRnn.initialize();
  const ns = NoteSequence.create({
    ticksPerQuarter: 220,
    totalTime: 1.5,
    timeSignatures: [
      NoteSequence.TimeSignature.create({
        time: 0,
        numerator: 4,
        denominator: 4
      })
    ],
    tempos: [
      NoteSequence.Tempo.create({
        time: 0,
        qpm: 120
      })
    ],
    notes: [
      NoteSequence.Note.create({
        instrument: 0,
        program: 0,
        startTime: 0,
        endTime: 0.5,
        pitch: 60,
        velocity: 100,
        isDrum: false
      }),
      NoteSequence.Note.create({
        instrument: 0,
        program: 0,
        startTime: 0.5,
        endTime: 1.0,
        pitch: 60,
        velocity: 100,
        isDrum: false
      }),
      NoteSequence.Note.create({
        instrument: 0,
        program: 0,
        startTime: 1.0,
        endTime: 1.5,
        pitch: 67,
        velocity: 100,
        isDrum: false
      }),
      NoteSequence.Note.create({
        instrument: 0,
        program: 0,
        startTime: 1.5,
        endTime: 2.0,
        pitch: 67,
        velocity: 100,
        isDrum: false
      }),
    ]
  });
  const qns = magenta.Sequences.quantizeNoteSequence(ns, 1);
  console.log(await melodyRnn.continueSequence(qns, 20));
}

runMelodyRnn();
