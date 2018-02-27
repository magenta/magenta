// tslint:disable-next-line:max-line-length
import { MusicVAE, Note, DrumsConverter, DrumRollConverter, MelodyConverter } from '../index';
import * as dl from 'deeplearn';

// tslint:disable:max-line-length
const DRUMS_CKPT = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/drums_small_hikl';
const DRUMS_NADE_CKPT = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/drums_nade_9';
const MEL_CKPT = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/mel_small';
// tslint:enable:max-line-length

const DRUM_SEQS = [
  [new Note(36, 0), new Note(42, 2), new Note(36, 4), new Note(42, 6),
   new Note(36, 8), new Note(42, 10), new Note(36, 12), new Note(42, 14),
   new Note(36, 16), new Note(36, 24), new Note(36, 28), new Note(42, 30)],
  [new Note(36, 0), new Note(38, 0), new Note(42, 0), new Note(46, 0),
   new Note(42, 2), new Note(42, 3), new Note(42, 4), new Note(50, 4),
   new Note(36, 6), new Note(38, 6), new Note(42, 6), new Note(45, 6),
   new Note(36, 8), new Note(42, 8), new Note(46, 8), new Note(42, 10),
   new Note(48, 10), new Note(50, 10), new Note(36, 12), new Note(38, 12),
   new Note(42, 12), new Note(48, 12), new Note(50, 13), new Note(42, 14),
   new Note(45, 14), new Note(48, 14), new Note(36, 16), new Note(38, 16),
   new Note(42, 16), new Note(46, 16), new Note(49, 16), new Note(42, 18),
   new Note(42, 19), new Note(42, 20), new Note(50, 20), new Note(36, 22),
   new Note(38, 22), new Note(42, 22), new Note(45, 22), new Note(36, 24),
   new Note(42, 24), new Note(46, 24), new Note(42, 26), new Note(48, 26),
   new Note(50, 26), new Note(36, 28), new Note(38, 28), new Note(42, 28),
   new Note(42, 30), new Note(48, 30)],
  [new Note(38, 0), new Note(42, 0), new Note(42, 2), new Note(42, 4),
   new Note(36, 6), new Note(38, 6), new Note(42, 6), new Note(45, 6),
   new Note(36, 8), new Note(42, 8), new Note(42, 10), new Note(38, 12),
   new Note(42, 12), new Note(45, 12), new Note(36, 14), new Note(42, 14),
   new Note(46, 14), new Note(36, 16), new Note(42, 16), new Note(42, 18),
   new Note(38, 20), new Note(42, 20), new Note(45, 20), new Note(36, 22),
   new Note(42, 22), new Note(36, 24), new Note(42, 24), new Note(38, 26),
   new Note(42, 26), new Note(45, 26), new Note(42, 28), new Note(45, 28),
   new Note(36, 30), new Note(42, 30), new Note(45, 30)],
   [new Note(50, 4), new Note(50, 20)]];

function writeTimer(elementId: string, startTime: number) {
  document.getElementById(elementId).innerHTML = (
    (performance.now() - startTime) / 1000.).toString() + 's';
}

function writeNoteSeqs(elementId: string, seqs: Note[][]) {
  document.getElementById(elementId).innerHTML = seqs.map(
    seq => '[' + seq.map(n => n.toString()).join(', ') + ']').join('<br>');
}

async function runDrums(){
  const mvae: MusicVAE = new MusicVAE(DRUMS_CKPT, new DrumsConverter(32));
  await mvae.initialize();

  writeNoteSeqs('drums-inputs', DRUM_SEQS);

  let start = performance.now();

  const interp = mvae.interpolate(DRUM_SEQS, 3);
  writeTimer('drums-interp-time', start);
  writeNoteSeqs('drums-interp', interp);

  start = performance.now();
  const sample = mvae.sample(5);
  writeTimer('drums-sample-time', start);
  writeNoteSeqs('drums-samples', sample);

  mvae.dispose();
  console.log(dl.memory());
}

async function runDrumsNade(){
  const mvae: MusicVAE = new MusicVAE(
      DRUMS_NADE_CKPT, new DrumRollConverter(32));
  await mvae.initialize();

  writeNoteSeqs('nade-inputs', DRUM_SEQS);

  let start = performance.now();
  const interp = mvae.interpolate(DRUM_SEQS, 3);
  writeTimer('nade-interp-time', start);
  writeNoteSeqs('nade-interp', interp);

  start = performance.now();
  const sample = mvae.sample(5);
  writeTimer('nade-sample-time', start);
  writeNoteSeqs('nade-samples', sample);

  mvae.dispose();
  console.log(dl.memory());
}

async function runMel(){
  const mvae:MusicVAE = new MusicVAE(MEL_CKPT, new MelodyConverter(32));
  await mvae.initialize();

  const teaPot = [
      new Note(69, 0, 2), new Note(71, 2, 4), new Note(73, 4, 6),
      new Note(74, 6, 8), new Note(76, 8, 10), new Note(81, 12, 16),
      new Note(77, 16, 20), new Note(80, 20, 24), new Note(75, 24, 32)];

  const twinkle = [
      new Note(60, 0, 2), new Note(60, 2, 4), new Note(67, 4, 6),
      new Note(67, 6, 8), new Note(69, 8, 10), new Note(69, 10, 12),
      new Note(67, 12, 16), new Note(65, 16, 18), new Note(65, 18, 20),
      new Note(64, 20, 22), new Note(64, 22, 24), new Note(62, 24, 26),
      new Note(62, 26, 28), new Note(60, 28, 32)];

  writeNoteSeqs('mel-inputs', [teaPot, twinkle]);

  let start = performance.now();
  const interp = mvae.interpolate([teaPot, twinkle], 5);
  writeTimer('mel-interp-time', start);
  writeNoteSeqs('mel-interp', interp);

  start = performance.now();
  const sample = mvae.sample(5);
  writeTimer('mel-sample-time', start);
  writeNoteSeqs('mel-samples', sample);

  mvae.dispose();
  console.log(dl.memory());
}

try {
  runDrums();
  runDrumsNade();
  runMel();
} catch (err){
  console.error(err);
}
