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
import INoteSequence = tensorflow.magenta.INoteSequence;
import * as midiconvert from 'midiconvert';
import * as constants from './constants';

export class MidiConversionError extends Error {
  constructor(message?: string) {
    super(message);
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

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
    if (parsedMidi.header.timeSignature) {
      ns.timeSignatures.push(NoteSequence.TimeSignature.create({
        time: 0,
        numerator: parsedMidi.header.timeSignature[0],
        denominator: parsedMidi.header.timeSignature[1],
      }));
    } else {
      // Assume a default time signature of 4/4.
      ns.timeSignatures.push(NoteSequence.TimeSignature.create({
        time: 0,
        numerator: 4,
        denominator: 4,
      }));
    }

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

  public static SequenceProtoToMidi(ns:INoteSequence) {
    if (!ns.tempos || ns.tempos.length !== 1 || ns.tempos[0].time !== 0) {
      throw new MidiConversionError(
        'NoteSequence must have exactly 1 tempo at time 0');
    }
    if (!ns.timeSignatures || ns.timeSignatures.length !== 1 ||
      ns.timeSignatures[0].time !== 0) {
      throw new MidiConversionError(
        'NoteSequence must have exactly 1 time signature at time 0');
    }
    const json = {
      header: {
        bpm: ns.tempos[0].qpm,
        PPQ: ns.ticksPerQuarter,
        timeSignature: [
          ns.timeSignatures[0].numerator, ns.timeSignatures[0].denominator]
      },
      tracks: [] as Array<{}>
    };
    const tracks:{[instrument: number]: NoteSequence.INote[]} = {};
    for (const note of ns.notes) {
      const track = note.instrument;
      if (!(track in tracks)) {
        tracks[track] = [];
      }
      tracks[track].push(note);
    }
    const instruments = Object.keys(tracks).map(x => parseInt(x, 10)).sort();
    for (let i = 0; i < instruments.length; i++) {
      if (i !== instruments[i]) {
        throw new MidiConversionError(
          'Instrument list must be continuous and start at 0');
      }

      const track = {
        id: i,
        notes: [] as Array<{}>,
        isPercussion: tracks[i][0].isDrum,
        channelNumber: i,
        instrumentNumber: tracks[i][0].program
      };

      for (const note of tracks[i]) {
        track.notes.push({
          midi: note.pitch,
          time: note.startTime,
          duration: note.endTime - note.startTime,
          velocity: (note.velocity as number + 1) / constants.MIDI_VELOCITIES
        });
      }

      json['tracks'].push(track);
    }

    return midiconvert.fromJSON(json).encode();
  }
}
