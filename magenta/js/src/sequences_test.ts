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
// tslint:disable-next-line:no-require-imports Generated protobuf code.
const music_pb = require('./music_pb');
import {Sequences} from './sequences';

const STEPS_PER_QUARTER = 4;

function createTestNS() {
    var ns = new music_pb.NoteSequence();

    var tempo = new music_pb.NoteSequence.Tempo();
    tempo.setQpm(60);
    tempo.setTime(0);
    ns.addTempos(tempo);

    var ts = new music_pb.NoteSequence.TimeSignature();
    ts.setTime(0);
    ts.setNumerator(4);
    ts.setDenominator(4);
    ns.addTimeSignatures(ts);

    return ns;
}

function addTrackToSequence(ns, instrument, notes) {
    for (var noteParams of notes) {
        var note = new music_pb.NoteSequence.Note();
        note.setPitch(noteParams[0]);
        note.setVelocity(noteParams[1]);
        note.setStartTime(noteParams[2]);
        note.setEndTime(noteParams[3]);
        ns.addNotes(note);
        if(ns.getTotalTime() < note.getEndTime()) {
            ns.setTotalTime(note.getEndTime());
        }
    }
}

function addChordsToSequence(ns, chords) {
    for (var chordParams of chords) {
        var ta = new music_pb.NoteSequence.TextAnnotation();
        ta.setText(chordParams[0]);
        ta.setTime(chordParams[1]);
        ta.setAnnotationType(
            music_pb.NoteSequence.TextAnnotation.TextAnnotationType.CHORD_SYMBOL);
        ns.addTextAnnotations(ta);
    }
}

function addControlChangesToSequence(ns, instrument, chords) {
    for (var ccParams of chords) {
        var cc = new music_pb.NoteSequence.ControlChange();
        cc.setTime(ccParams[0]);
        cc.setControlNumber(ccParams[1]);
        cc.setControlValue(ccParams[2]);
        cc.setInstrument(instrument);
        ns.addControlChanges(cc);
    }
}

function addQuantizedStepsToSequence(ns, quantizedSteps) {
    quantizedSteps.forEach(function(qstep, i) {
        var note = ns.getNotesList()[i];
        note.setQuantizedStartStep(qstep[0]);
        note.setQuantizedEndStep(qstep[1]);
        if(note.getQuantizedEndStep() > ns.getTotalQuantizedSteps()) {
            ns.setTotalQuantizedSteps(note.getQuantizedEndStep());
        }
    });
}

function addQuantizedChordStepsToSequence(ns, quantizedSteps) {
    var chordAnnotations = ns.getTextAnnotationsList().filter(
        ta => ta.getAnnotationType() ==
        music_pb.NoteSequence.TextAnnotation.TextAnnotationType.CHORD_SYMBOL);
    
    quantizedSteps.forEach(function(qstep, i) {
        var ta = chordAnnotations[i];
        ta.setQuantizedStep(qstep);
    });
}

function addQuantizedControlStepsToSequence(ns, quantizedSteps) {
    quantizedSteps.forEach(function(qstep, i) {
        var cc = ns.getControlChangesList()[i];
        cc.setQuantizedStep(qstep);
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
