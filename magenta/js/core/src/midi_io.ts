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
import * as midiconvert from 'midiconvert';
import * as constants from './constants';

export class MidiIO {
  public static midiToSequenceProto(midi: string): NoteSequence {
    const parsedMidi = midiconvert.parse(midi);
    const ns = NoteSequence.create();

    ns.ticksPerQuarter = parsedMidi.header.PPQ;
    ns.sourceInfo = NoteSequence.SourceInfo.create({
      parser: NoteSequence.SourceInfo.Parser.TONEJS_MIDI_CONVERT,
      encodingType: NoteSequence.SourceInfo.EncodingType.MIDI
    });

    // TODO(fjord): When MidiConvert supports multiple time signatures, update
    // accordingly.
    ns.timeSignatures.push(NoteSequence.TimeSignature.create({
      time: 0,
      numerator: parsedMidi.header.timeSignature[0],
      denominator: parsedMidi.header.timeSignature[1],
    }));

    // TODO(fjord): Add key signatures when MidiConvert supports them.

    // TODO(fjord): When MidiConvert supports multiple tempos, update
    // accordingly.
    ns.tempos.push(NoteSequence.Tempo.create({
      time: 0,
      qpm: parsedMidi.header.bpm
    }));

    // We want a unique instrument number for each combination of track and
    // program number.
    let instrumentNumber = -1;
    for (const track of parsedMidi.tracks) {
      // TODO(fjord): support changing programs within a track when
      // MidiConvert does. When that happens, we'll need a map to keep track
      // of which program number within a track corresponds to what instrument
      // number, similar to how pretty_midi works.
      if (track.notes.length > 0) {
        instrumentNumber += 1;
      }

      for (const note of track.notes) {
        const startTime:number = note.time;
        const duration:number = note.duration;
        const endTime:number = startTime + duration;

        ns.notes.push(NoteSequence.Note.create({
          instrument: instrumentNumber,
          program: track.instrumentNumber,
          startTime,
          endTime,
          pitch: note.midi,
          velocity: Math.floor(note.velocity * constants.MIDI_VELOCITIES),
          isDrum: track.isPercussion
        }));

        if (endTime > ns.totalTime) {
          ns.totalTime = endTime;
        }
      }
    }

    return ns;
  }
}
