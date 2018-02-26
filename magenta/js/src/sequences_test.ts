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
import music_pb = require('./music_pb');
import {Sequences} from './sequences';

const STEPS_PER_QUARTER = 4;

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

test("Quantize NoteSequence", (t:test.Test) => {
    var ns = new music_pb.NoteSequence();

    addTrackToSequence(
        ns, 0,
        [[12, 100, 0.01, 10.0], [11, 55, 0.22, 0.50], [40, 45, 2.50, 3.50],
        [55, 120, 4.0, 4.01], [52, 99, 4.75, 5.0]]);
    // testing_lib.add_chords_to_sequence(
    //     self.note_sequence,
    //     [('B7', 0.22), ('Em9', 4.0)])
    // testing_lib.add_control_changes_to_sequence(
    //     self.note_sequence, 0,
    //     [(2.0, 64, 127), (4.0, 64, 0)])

    var expectedQuantizedSequence = ns.clone();
    // expected_quantized_sequence.quantization_info.steps_per_quarter = (
    //     self.steps_per_quarter)
    // testing_lib.add_quantized_steps_to_sequence(
    //     expected_quantized_sequence,
    //     [(0, 40), (1, 2), (10, 14), (16, 17), (19, 20)])
    // testing_lib.add_quantized_chord_steps_to_sequence(
    //     expected_quantized_sequence, [1, 16])
    // testing_lib.add_quantized_control_steps_to_sequence(
    //     expected_quantized_sequence, [8, 16])

    var qns = Sequences.quantizeNoteSequence(ns, STEPS_PER_QUARTER);
    t.true(qns);

    // self.assertProtoEquals(expected_quantized_sequence, quantized_sequence)
    t.end();
});