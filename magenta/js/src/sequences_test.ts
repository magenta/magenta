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
import {Sequences} from './sequences';

const STEPS_PER_QUARTER = 4;

import NoteSequence = tensorflow.magenta.NoteSequence;

function createTestNS() {
    var ns = NoteSequence.create();

    ns.tempos.push(NoteSequence.Tempo.create({qpm: 60, time: 0}));

    ns.timeSignatures.push(NoteSequence.TimeSignature.create({
        time: 0,
        numerator: 4,
        denominator: 4,
    }));

    return ns;
}

function addTrackToSequence(ns: NoteSequence, instrument: number,
        notes: Array<Array<number>>) {
    for (var noteParams of notes) {
        var note = new NoteSequence.Note({
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

function addChordsToSequence(ns: NoteSequence, chords: Array<Array<number|string>>) {
    for (var chordParams of chords) {
        var ta = NoteSequence.TextAnnotation.create({
            text: chordParams[0] as string,
            time: chordParams[1] as number,
            annotationType: 
                NoteSequence.TextAnnotation.TextAnnotationType.CHORD_SYMBOL
        });
        ns.textAnnotations.push(ta);
    }
}

function addControlChangesToSequence(ns: NoteSequence, instrument:number, chords:Array<Array<number>>) {
    for (var ccParams of chords) {
        var cc = NoteSequence.ControlChange.create({
            time: ccParams[0],
            controlNumber: ccParams[1],
            controlValue: ccParams[2],
            instrument: instrument
        });
        ns.controlChanges.push(cc);
    }
}

function addQuantizedStepsToSequence(ns: NoteSequence, quantizedSteps: Array<Array<number>>) {
    quantizedSteps.forEach(function(qstep, i) {
        var note = ns.notes[i];
        note.quantizedStartStep = qstep[0];
        note.quantizedEndStep = qstep[1];
        if(note.quantizedEndStep > ns.totalQuantizedSteps) {
            ns.totalQuantizedSteps = note.quantizedEndStep;
        }
    });
}

function addQuantizedChordStepsToSequence(ns:NoteSequence, quantizedSteps:Array<number>) {
    var chordAnnotations = ns.textAnnotations.filter(
        ta => ta.annotationType ==
        NoteSequence.TextAnnotation.TextAnnotationType.CHORD_SYMBOL);
    
    quantizedSteps.forEach(function(qstep, i) {
        var ta = chordAnnotations[i];
        ta.quantizedStep = qstep;
    });
}

function addQuantizedControlStepsToSequence(ns:NoteSequence, quantizedSteps:Array<number>) {
    quantizedSteps.forEach(function(qstep, i) {
        var cc = ns.controlChanges[i];
        cc.quantizedStep = qstep;
    });
}

test("Quantize NoteSequence", (t:test.Test) => {
    var ns = createTestNS();

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

    var expectedQuantizedSequence = ns.clone();
    expectedQuantizedSequence.setQuantizationInfo(
        new music_pb.NoteSequence.QuantizationInfo());
    expectedQuantizedSequence.getQuantizationInfo().setStepsPerQuarter(
        STEPS_PER_QUARTER);
    addQuantizedStepsToSequence(
        expectedQuantizedSequence,
        [[0, 40], [1, 2], [10, 14], [16, 17], [19, 20]]);
    addQuantizedChordStepsToSequence(
        expectedQuantizedSequence, [1, 16]);
    addQuantizedControlStepsToSequence(
        expectedQuantizedSequence, [8, 16]);

    var qns = Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);

    t.deepEqual(qns.toObject(), expectedQuantizedSequence.toObject());

    t.end();
});
