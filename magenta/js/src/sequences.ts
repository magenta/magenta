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

export class Sequences {
    /**
     * Quantize a NoteSequence proto relative to tempo.
     *
     * The input NoteSequence is copied and quantization-related fields are
     * populated. Sets the `steps_per_quarter` field in the `quantization_info`
     * message in the NoteSequence.

     * Note start and end times, and chord times are snapped to a nearby quantized
     * step, and the resulting times are stored in a separate field (e.g.,
     * quantized_start_step). See the comments above `QUANTIZE_CUTOFF` for details on
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

        // if qns.time_signatures:
        //     time_signatures = sorted(qns.time_signatures, key=lambda ts: ts.time)
        //     # There is an implicit 4/4 time signature at 0 time. So if the first time
        //     # signature is something other than 4/4 and it's at a time other than 0,
        //     # that's an implicit time signature change.
        //     if time_signatures[0].time != 0 and not (
        //         time_signatures[0].numerator == 4 and
        //         time_signatures[0].denominator == 4):
        //     raise MultipleTimeSignatureException(
        //         'NoteSequence has an implicit change from initial 4/4 time '
        //         'signature to %d/%d at %.2f seconds.' % (
        //             time_signatures[0].numerator, time_signatures[0].denominator,
        //             time_signatures[0].time))

        //     for time_signature in time_signatures[1:]:
        //     if (time_signature.numerator != qns.time_signatures[0].numerator or
        //         time_signature.denominator != qns.time_signatures[0].denominator):
        //         raise MultipleTimeSignatureException(
        //             'NoteSequence has at least one time signature change from %d/%d to '
        //             '%d/%d at %.2f seconds.' % (
        //                 time_signatures[0].numerator, time_signatures[0].denominator,
        //                 time_signature.numerator, time_signature.denominator,
        //                 time_signature.time))

        //     # Make it clear that there is only 1 time signature and it starts at the
        //     # beginning.
        //     qns.time_signatures[0].time = 0
        //     del qns.time_signatures[1:]
        // else:
        //     time_signature = qns.time_signatures.add()
        //     time_signature.numerator = 4
        //     time_signature.denominator = 4
        //     time_signature.time = 0

        // if not _is_power_of_2(qns.time_signatures[0].denominator):
        //     raise BadTimeSignatureException(
        //         'Denominator is not a power of 2. Time signature: %d/%d' %
        //         (qns.time_signatures[0].numerator, qns.time_signatures[0].denominator))

        // if qns.time_signatures[0].numerator == 0:
        //     raise BadTimeSignatureException(
        //         'Numerator is 0. Time signature: %d/%d' %
        //         (qns.time_signatures[0].numerator, qns.time_signatures[0].denominator))

        // if qns.tempos:
        //     tempos = sorted(qns.tempos, key=lambda t: t.time)
        //     # There is an implicit 120.0 qpm tempo at 0 time. So if the first tempo is
        //     # something other that 120.0 and it's at a time other than 0, that's an
        //     # implicit tempo change.
        //     if tempos[0].time != 0 and (
        //         tempos[0].qpm != constants.DEFAULT_QUARTERS_PER_MINUTE):
        //     raise MultipleTempoException(
        //         'NoteSequence has an implicit tempo change from initial %.1f qpm to '
        //         '%.1f qpm at %.2f seconds.' % (
        //             constants.DEFAULT_QUARTERS_PER_MINUTE, tempos[0].qpm,
        //             tempos[0].time))

        //     for tempo in tempos[1:]:
        //     if tempo.qpm != qns.tempos[0].qpm:
        //         raise MultipleTempoException(
        //             'NoteSequence has at least one tempo change from %.1f qpm to %.1f '
        //             'qpm at %.2f seconds.' % (tempos[0].qpm, tempo.qpm, tempo.time))

        //     # Make it clear that there is only 1 tempo and it starts at the beginning.
        //     qns.tempos[0].time = 0
        //     del qns.tempos[1:]
        // else:
        //     tempo = qns.tempos.add()
        //     tempo.qpm = constants.DEFAULT_QUARTERS_PER_MINUTE
        //     tempo.time = 0

        // # Compute quantization steps per second.
        // steps_per_second = steps_per_quarter_to_steps_per_second(
        //     steps_per_quarter, qns.tempos[0].qpm)

        // qns.total_quantized_steps = quantize_to_step(qns.total_time, steps_per_second)
        // _quantize_notes(qns, steps_per_second)

        // return qns
        return qns
    }
}