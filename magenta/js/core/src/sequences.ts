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

import { tensorflow } from '@magenta/protobuf';
import NoteSequence = tensorflow.magenta.NoteSequence;
import * as constants from './constants';

// Set the quantization cutoff.
// Note events before this cutoff are rounded down to nearest step. Notes
// above this cutoff are rounded up to nearest step. The cutoff is given as a
// fraction of a step.
// For example, with quantize_cutoff = 0.75 using 0-based indexing,
// if .75 < event <= 1.75, it will be quantized to step 1.
// If 1.75 < event <= 2.75 it will be quantized to step 2.
// A number close to 1.0 gives less wiggle room for notes that start early,
// and they will be snapped to the previous step.
const QUANTIZE_CUTOFF = 0.5;

export class MultipleTimeSignatureException extends Error {
  constructor(message?: string) {
    super(message);
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
export class BadTimeSignatureException extends Error {
  constructor(message?: string) {
    super(message);
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
export class NegativeTimeException extends Error {
  constructor(message?: string) {
    super(message);
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
export class MultipleTempoException extends Error {
  constructor(message?: string) {
    super(message);
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class Sequences {
  private static isPowerOf2(n: number): boolean {
    return n && (n & (n - 1)) === 0;
  }

  /**
   * Calculates steps per second given stepsPerQuarter and a QPM.
   */
  public static stepsPerQuarterToStepsPerSecond(stepsPerQuarter: number,
    qpm: number): number {
    return stepsPerQuarter * qpm / 60.0;
  }

  /**
   * Quantizes seconds to the nearest step, given steps_per_second.
   * See the comments above `QUANTIZE_CUTOFF` for details on how the
   * quantizing algorithm works.
   * @param unquantizedSeconds Seconds to quantize.
   * @param stepsPerSecond Quantizing resolution.
   * @param quantizeCutoff Value to use for quantizing cutoff.
   * @returns the quantized step.
   */
  public static quantizeToStep(unquantizedSeconds: number,
    stepsPerSecond: number, quantizeCutoff = QUANTIZE_CUTOFF): number {
    const unquantizedSteps = unquantizedSeconds * stepsPerSecond;
    return Math.floor(unquantizedSteps + (1 - quantizeCutoff));
  }

  /**
   * Quantize the notes and chords of a NoteSequence proto in place.
   * Note start and end times, and chord times are snapped to a nearby
   * quantized step, and the resulting times are stored in a separate field
   * (e.g. QuantizedStartStep). See the comments above `QUANTIZE_CUTOFF` for
   * details on how the quantizing algorithm works.
   * @param ns A music_pb2.NoteSequence protocol buffer. Will be modified in
   *    place.
   * @param stepsPerSecond Each second will be divided into this many
   *    quantized time steps.
   */
  private static quantizeNotes(ns: NoteSequence, stepsPerSecond: number) {
    for (const note of ns.notes) {
      // Quantize the start and end times of the note.
      note.quantizedStartStep = this.quantizeToStep(
        note.startTime, stepsPerSecond);
      note.quantizedEndStep = this.quantizeToStep(
        note.endTime, stepsPerSecond);
      if (note.quantizedEndStep === note.quantizedStartStep) {
        note.quantizedEndStep += 1;
      }

      // Do not allow notes to start or end in negative time.
      if (note.quantizedStartStep < 0 || note.quantizedEndStep < 0) {
        throw new NegativeTimeException(
          `Got negative note time: start_step = ` +
          `${note.quantizedStartStep}, end_step = ` +
          `${note.quantizedEndStep}`);
      }

      // Extend quantized sequence if necessary.
      if (note.quantizedEndStep > ns.totalQuantizedSteps) {
        ns.totalQuantizedSteps = note.quantizedEndStep;
      }
    }

    // Also quantize control changes and text annotations.
    for (const event of ns.controlChanges.concat(ns.textAnnotations)) {
      // Quantize the event time, disallowing negative time.
      event.quantizedStep = this.quantizeToStep(event.time, stepsPerSecond);
      if (event.quantizedStep < 0) {
        throw new NegativeTimeException(
          `Got negative event time: step = ${event.quantizedStep}`);
      }
    }
  }

  /**
   * Quantize a NoteSequence proto relative to tempo.
   *
   * The input NoteSequence is copied and quantization-related fields are
   * populated. Sets the `steps_per_quarter` field in the `quantization_info`
   * message in the NoteSequence.
   *
   * Note start and end times, and chord times are snapped to a nearby quantized
   * step, and the resulting times are stored in a separate field (e.g.,
   * QuantizedStartStep). See the comments above `QUANTIZE_CUTOFF` for details
   * on how the quantizing algorithm works.
   *
   * @param note_sequence A music_pb2.NoteSequence protocol buffer.
   * @param steps_per_quarter Each quarter note of music will be divided into
   *    this many quantized time steps.
   * @returns A copy of the original NoteSequence, with quantized times added.
   *
   * @throws {MultipleTempoException} If there is a change in tempo in
   *    note sequence.
   * @throws {MultipleTimeSignatureException} If there is a change in time
   *    signature in noteSequence.
   * @throws {BadTimeSignatureException} If the time signature found in
   *    noteSequence has a 0 numerator or a denominator which is not a power of
   *    2.
   * @throws {NegativeTimeException} If a note or chord occurs at a negative
   *    time.
   */
  public static quantizeNoteSequence(noteSequence: NoteSequence,
    stepsPerQuarter: number): NoteSequence {
    // Make a copy.
    const qns = NoteSequence.fromObject(
      NoteSequence.toObject(noteSequence));

    qns.quantizationInfo = NoteSequence.QuantizationInfo.create({
      stepsPerQuarter
    });

    if (qns.timeSignatures.length > 0) {
      qns.timeSignatures.sort((a, b) => {
        return a.time - b.time;
      });
      // There is an implicit 4/4 time signature at 0 time. So if the first time
      // signature is something other than 4/4 and it's at a time other than 0,
      // that's an implicit time signature change.
      if (qns.timeSignatures[0].time !== 0 && !(
        qns.timeSignatures[0].numerator === 4 &&
        qns.timeSignatures[0].denominator === 4)) {
        throw new MultipleTimeSignatureException(
          'NoteSequence has an implicit change from initial 4/4 time ' +
          `signature to ${qns.timeSignatures[0].numerator}/` +
          `${qns.timeSignatures[0].denominator} at ` +
          `${qns.timeSignatures[0].time} seconds.`);
      }

      for (let i = 1; i < qns.timeSignatures.length; i++) {
        const timeSignature = qns.timeSignatures[i];
        if (timeSignature.numerator !== qns.timeSignatures[0].numerator ||
          timeSignature.denominator !== qns.timeSignatures[0].denominator) {
          throw new MultipleTimeSignatureException(
            'NoteSequence has at least one time signature change from ' +
            `${qns.timeSignatures[0].numerator}/` +
            `${qns.timeSignatures[0].denominator} to ` +
            `${timeSignature.numerator}/${timeSignature.denominator} ` +
            `at ${timeSignature.time} seconds`);
        }
      }

      // Make it clear that there is only 1 time signature and it starts at the
      // beginning.
      qns.timeSignatures[0].time = 0;
      qns.timeSignatures = [qns.timeSignatures[0]];
    } else {
      const timeSignature = NoteSequence.TimeSignature.create({
        numerator: 4,
        denominator: 4,
        time: 0
      });
      qns.timeSignatures.push(timeSignature);
    }

    const firstTS = qns.timeSignatures[0];
    if (!this.isPowerOf2(firstTS.denominator)) {
      throw new BadTimeSignatureException(
        'Denominator is not a power of 2. Time signature: ' +
        `${firstTS.numerator}/${firstTS.denominator}`);
    }

    if (firstTS.numerator === 0) {
      throw new BadTimeSignatureException(
        'Numerator is 0. Time signature: ' +
        `${firstTS.numerator}/${firstTS.denominator}`);
    }

    if (qns.tempos.length > 0) {
      qns.tempos.sort((a, b) => {
        return a.time - b.time;
      });
      // There is an implicit 120.0 qpm tempo at 0 time. So if the first tempo
      // is something other that 120.0 and it's at a time other than 0, that's
      // an implicit tempo change.
      if (qns.tempos[0].time !== 0 &&
        qns.tempos[0].qpm !== constants.DEFAULT_QUARTERS_PER_MINUTE) {
        throw new MultipleTempoException(
          'NoteSequence has an implicit tempo change from initial ' +
          `${constants.DEFAULT_QUARTERS_PER_MINUTE} qpm to ` +
          `${qns.tempos[0].qpm} qpm at ${qns.tempos[0].time} seconds.`);
      }

      for (let i = 1; i < qns.tempos.length; i++) {
        if (qns.tempos[i].qpm !== qns.tempos[0].qpm) {
          throw new MultipleTempoException(
            'NoteSequence has at least one tempo change from ' +
            `${qns.tempos[0].qpm} qpm to ${qns.tempos[i].qpm}` +
            `qpm at ${qns.tempos[i].time} seconds.`);
        }
      }

      // Make it clear that there is only 1 tempo and it starts at the beginning
      qns.tempos[0].time = 0;
      qns.tempos = [qns.tempos[0]];
    } else {
      const tempo = NoteSequence.Tempo.create({
        qpm: constants.DEFAULT_QUARTERS_PER_MINUTE,
        time: 0
      });
      qns.tempos.push(tempo);
    }

    // Compute quantization steps per second.
    const stepsPerSecond = this.stepsPerQuarterToStepsPerSecond(
      stepsPerQuarter, qns.tempos[0].qpm);

    qns.totalQuantizedSteps = this.quantizeToStep(
      qns.totalTime, stepsPerSecond);
    this.quantizeNotes(qns, stepsPerSecond);

    // return qns
    return qns;
  }
}
