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
import INoteSequence = magenta.INoteSequence;
import { MelodyRnn } from '../src/index';
import * as dl from 'deeplearn';

const MELODY_NS:INoteSequence = {
  ticksPerQuarter: 220,
  totalTime: 1.5,
  timeSignatures: [
    {
      time: 0,
      numerator: 4,
      denominator: 4
    }
  ],
  tempos: [
    {
      time: 0,
      qpm: 120
    }
  ],
  notes: [
    {
      instrument: 0,
      program: 0,
      startTime: 0,
      endTime: 0.5,
      pitch: 60,
      velocity: 100,
      isDrum: false
    },
    {
      instrument: 0,
      program: 0,
      startTime: 0.5,
      endTime: 1.0,
      pitch: 60,
      velocity: 100,
      isDrum: false
    },
    {
      instrument: 0,
      program: 0,
      startTime: 1.0,
      endTime: 1.5,
      pitch: 67,
      velocity: 100,
      isDrum: false
    },
    {
      instrument: 0,
      program: 0,
      startTime: 1.5,
      endTime: 2.0,
      pitch: 67,
      velocity: 100,
      isDrum: false
    },
  ]
};

function writeTimer(elementId: string, startTime: number) {
  document.getElementById(elementId).innerHTML = (
  (performance.now() - startTime) / 1000.).toString() + 's';
}

function writeNoteSeqs(elementId: string, seqs: INoteSequence[]) {
  document.getElementById(elementId).innerHTML = seqs.map(
  seq => '[' + seq.notes.map(n => {
    let s = '{p:' + n.pitch + ' s:' + n.quantizedStartStep;
    if (n.quantizedEndStep != null) {
      s += ' e:' +  n.quantizedEndStep;
    }
    s += '}';
    return s;
  }).join(', ') + ']').join('<br>');
}

async function runMelodyRnn() {
  const melodyRnn = new MelodyRnn();
  await melodyRnn.initialize();

  const qns = magenta.Sequences.quantizeNoteSequence(MELODY_NS, 1);
  writeNoteSeqs('melody-cont-inputs', [qns]);
  const start = performance.now();
  const continuation = await melodyRnn.continueSequence(qns, 20);
  writeTimer('melody-cont-time', start);
  writeNoteSeqs('melody-cont-results', [continuation]);
  melodyRnn.dispose();

  console.log(dl.getBackend());
  console.log(dl.memory());
}


runMelodyRnn();
