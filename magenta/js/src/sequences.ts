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

import music_pb = require('./music_pb');
import * as constants from './constants';

class MultipleTimeSignatureException extends Error {}
class BadTimeSignatureException extends Error {}
class NegativeTimeException extends Error {}
class MultipleTempoException extends Error {}

// Set the quantization cutoff.
// Note events before this cutoff are rounded down to nearest step. Notes
// above this cutoff are rounded up to nearest step. The cutoff is given as a
// fraction of a step.
// For example, with quantize_cutoff = 0.75 using 0-based indexing,
// if .75 < event <= 1.75, it will be quantized to step 1.
// If 1.75 < event <= 2.75 it will be quantized to step 2.
// A number close to 1.0 gives less wiggle room for notes that start early,
// and they will be snapped to the previous step.
const QUANTIZE_CUTOFF = 0.5

export class Sequences {
    private static isPowerOf2(n) {
        return n && (n & (n - 1)) === 0;
    }

    /**
     * Calculates steps per second given stepsPerQuarter and a QPM.
     */
    public static stepsPerQuarterToStepsPerSecond(stepsPerQuarter, qpm) {
        return stepsPerQuarter * qpm / 60.0;
    }

    /**
     * Quantizes seconds to the nearest step, given steps_per_second.
     * See the comments above `QUANTIZE_CUTOFF` for details on how the quantizing
     * algorithm works.
     * @param unquantizedSeconds Seconds to quantize.
     * @param stepsPerSecond Quantizing resolution.
     * @param quantizeCutoff Value to use for quantizing cutoff.
     */
    public static quantizeToStep(unquantizedSeconds, stepsPerSecond,
            quantizeCutoff=QUANTIZE_CUTOFF) {
        var unquantizedSteps = unquantizedSeconds * stepsPerSecond;
        return Math.floor(unquantizedSteps + (1 - quantizeCutoff));
    }

    /**
     * Quantize the notes and chords of a NoteSequence proto in place.

     * Note start and end times, and chord times are snapped to a nearby quantized
     * step, and the resulting times are stored in a separate field (e.g.,
     * QuantizedStartStep). See the comments above `QUANTIZE_CUTOFF` for details on
     * how the quantizing algorithm works.
     * @param ns A music_pb2.NoteSequence protocol buffer. Will be modified in place.
     * @param stepsPerSecond Each second will be divided into this many quantized time steps.
     */
    private static quantizeNotes(ns, stepsPerSecond) {
        for(var note of ns.getNotesList()) {
            // Quantize the start and end times of the note.
            note.setQuantizedStartStep(this.quantizeToStep(
                note.getStartTime(), stepsPerSecond));
            note.setQuantizedEndStep(this.quantizeToStep(
                note.getEndTime(), stepsPerSecond));
            if(note.getQuantizedEndStep() === note.getQuantizedStartStep()) {
                note.setQuantizedEndStep(note.getQuantizedEndStep() + 1);
            }

            // Do not allow notes to start or end in negative time.
            if(note.getQuantizedStartStep() < 0 || note.getQuantizedEndStep() < 0) {
                throw new NegativeTimeException(
                    'Got negative note time: start_step = ' +
                    note.getQuantizedStartStep() + ', end_step = ' +
                    note.getQuantizedEndStep());
            }

            // Extend quantized sequence if necessary.
            if(note.getQuantizedEndStep() > ns.getTotalQuantizedSteps()) {
                ns.setTotalQuantizedSteps(note.getQuantizedEndStep());
            }
        }

        // Also quantize control changes and text annotations.
        for(var event of ns.getControlChangesList().concat(ns.getTextAnnotationsList())) {
            // Quantize the event time, disallowing negative time.
            event.setQuantizedStep(this.quantizeToStep(event.getTime(), stepsPerSecond));
            if(event.getQuantizedStep() < 0) {
                throw new NegativeTimeException(
                    'Got negative event time: step = ' + event.getQuantizedStep());
            }
        }
    }

    /**
     * Quantize a NoteSequence proto relative to tempo.
     *
     * The input NoteSequence is copied and quantization-related fields are
     * populated. Sets the `steps_per_quarter` field in the `quantization_info`
     * message in the NoteSequence.

     * Note start and end times, and chord times are snapped to a nearby quantized
     * step, and the resulting times are stored in a separate field (e.g.,
     * QuantizedStartStep). See the comments above `QUANTIZE_CUTOFF` for details on
     * how the quantizing algorithm works.
     *
     * Args:
     *     note_sequence: A music_pb2.NoteSequence protocol buffer.
     *     steps_per_quarter: Each quarter note of music will be divided into this
     *         many quantized time steps.
     *
     * Returns:
     *     A copy of the original NoteSequence, with quantized times added.
     *
     * Raises:
     *     MultipleTimeSignatureException: If there is a change in time signature
     *         in `note_sequence`.
     *     MultipleTempoException: If there is a change in tempo in `note_sequence`.
     *     BadTimeSignatureException: If the time signature found in `note_sequence`
     *         has a 0 numerator or a denominator which is not a power of 2.
     *     NegativeTimeException: If a note or chord occurs at a negative time.
     */
    public static quantizeNoteSequence(noteSequence: any, stepsPerQuarter: number): any {
        var qns = noteSequence.clone();

        qns.setQuantizationInfo(new music_pb.NoteSequence.QuantizationInfo())
        qns.getQuantizationInfo().setStepsPerQuarter(stepsPerQuarter);

        if(qns.getTimeSignaturesList().length > 0) {
            var timeSignatures = qns.getTimeSignaturesList();
            timeSignatures.sort(function(a, b) {
                return a.getTime() - b.getTime();
            });
            // There is an implicit 4/4 time signature at 0 time. So if the first time
            // signature is something other than 4/4 and it's at a time other than 0,
            // that's an implicit time signature change.
            if(timeSignatures[0].getTime() != 0 && !(
                    timeSignatures[0].getNumerator() == 4 &&
                    timeSignatures[0].getDenominator() == 4)) {
                throw new MultipleTimeSignatureException(
                    'NoteSequence has an implicit change from initial 4/4 time ' +
                    'signature to ' + timeSignatures[0].getNumerator() + '/' +
                    timeSignatures[0].getDenominator() + ' at ' +
                    timeSignatures[0].getTime() + ' seconds.');
            }

            for(var i = 1; i < timeSignatures.length; i++) {
                var timeSignature = timeSignatures[i];
                if (timeSignature.getNumerator() != timeSignatures[0].getNumerator() ||
                        timeSignature.getDenominator() != timeSignatures[0].getDenominator()) {
                    throw new MultipleTimeSignatureException(
                        'NoteSequence has at least one time signature change from ' +
                        timeSignatures[0].getNumerator() + '/' + timeSignatures[0].getDenominator() +
                        ' to ' + timeSignature.getNumerator() + '/' + timeSignature.getDenominator() +
                        'at ' + timeSignature.getTime() + ' seconds');
                }
            }

            // Make it clear that there is only 1 time signature and it starts at the
            // beginning.
            timeSignatures[0].setTime(0);
            qns.setTimeSignaturesList([timeSignatures[0]]);
        } else {
            var timeSignature = new music_pb.NoteSequence.TimeSignature();
            timeSignature.setNumerator(4);
            timeSignature.setDenominator(4);
            timeSignature.setTime(0);
            qns.addTimeSignatures(timeSignature);
        }

        var firstTS = qns.getTimeSignaturesList()[0];
        if(!this.isPowerOf2(firstTS.getDenominator())) {
            throw new BadTimeSignatureException(
                'Denominator is not a power of 2. Time signature: ' +
                firstTS.getNumerator() + '/' + firstTS.getDenominator());
        }

        if(firstTS.getNumerator() == 0) {
            throw new BadTimeSignatureException(
                'Numerator is 0. Time signature: ' +
                firstTS.getNumerator() + '/' + firstTS.getDenominator());
        }

        if(qns.getTemposList().length > 0) {
            var tempos = qns.getTemposList();
            tempos.sort(function(a, b) {
                return a.getTime() - b.getTime();
            });
            // There is an implicit 120.0 qpm tempo at 0 time. So if the first tempo is
            // something other that 120.0 and it's at a time other than 0, that's an
            // implicit tempo change.
            if(tempos[0].getTime() != 0 && (
                    tempos[0].getQpm() != constants.DEFAULT_QUARTERS_PER_MINUTE)) {
                throw new MultipleTempoException(
                    'NoteSequence has an implicit tempo change from initial ' +
                    constants.DEFAULT_QUARTERS_PER_MINUTE + ' qpm to ' +
                    tempos[0].getQpm() + ' qpm at ' + tempos[0].getTime() + ' seconds.');
            }

            for(var i = 1; i < tempos.length; i++) {
                if(tempos[i].getQpm() != qns.tempos[0].getQpm()) {
                    throw new MultipleTempoException(
                        'NoteSequence has at least one tempo change from ' +
                        tempos[0].getQpm() + ' qpm to ' + tempos[i].getQpm() +
                        'qpm at ' + tempos[i].getTime() + ' seconds.');
                }
            }

            // Make it clear that there is only 1 tempo and it starts at the beginning.
            tempos[0].setTime(0);
            qns.setTemposList([tempos[0]]);
        } else {
            var tempo = new music_pb.NoteSequence.Tempo();
            tempo.setQpm(constants.DEFAULT_QUARTERS_PER_MINUTE);
            tempo.setTime(0);
            qns.addTempos(tempo);
        }

        // Compute quantization steps per second.
        var stepsPerSecond = this.stepsPerQuarterToStepsPerSecond(
            stepsPerQuarter, qns.getTemposList()[0].getQpm());

        qns.setTotalQuantizedSteps(this.quantizeToStep(
            qns.getTotalTime(), stepsPerSecond));
        this.quantizeNotes(qns, stepsPerSecond);

        // return qns
        return qns
    }
}