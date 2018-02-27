import { MusicVAE, intsToBits, bitsToInts, intsToOneHot } from '../index';
import * as dl from 'deeplearn';

// tslint:disable:max-line-length
const DRUMS_CKPT = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/drums_small_hikl';
const DRUMS_NADE_CKPT = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/drums_nade_9';
const MEL_CKPT = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/mel_small';
// tslint:enable:max-line-length

function writeTimer(elementId: string, startTime: number) {
  document.getElementById(elementId).innerHTML = (
    (performance.now() - startTime) / 1000.).toString() + 's';
}

function write2dArray(elementId: string, arr: number[][]) {
  document.getElementById(elementId).innerHTML = arr.map(
      r => r.toString()).join('<br>');
}

async function runDrums(){
  const mvae: MusicVAE = new MusicVAE(DRUMS_CKPT);
  await mvae.initialize();

  const drums = [
    [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
     0, 0, 0, 1, 0, 4, 0],
    [15, 0, 4, 4, 68, 0, 23, 0, 13, 0, 100, 0, 39, 64, 52, 0, 143, 0, 4, 4, 68,
     0, 23, 0, 13, 0, 100, 0, 7, 0, 36, 0],
    [6, 0, 4, 0, 4, 0, 23, 0, 5, 0, 4, 0, 22, 0, 13, 0, 5, 0, 4, 0, 22, 0, 5, 0,
     5, 0, 22, 0, 20, 0, 21, 0],
    [0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0]];

  const drumsInput: number[][][] =
      [intsToBits(drums[0], 10), intsToBits(drums[1], 10),
       intsToBits(drums[2], 10), intsToBits(drums[3], 10)];

  document.getElementById('drums-inputs').innerHTML = drums.map(
      d => d.toString()).join('<br>');

  let start = performance.now();

  dl.tidy(() => {
    const interp = mvae.interpolate(dl.tensor3d(drumsInput), 3);
    const interpResults = [];
    for (let i = 0; i < interp.shape[0]; i++) {
      const bits: Uint8Array[] = [];
      for (let j = 0; j < interp.shape[1]; j++) {
        const r: dl.Array3D  = interp.slice([i, j, 0], [1, 1, interp.shape[2]]);
        bits.push(r.toInt().dataSync() as Uint8Array);
      }
      interpResults.push(bitsToInts(bits));
    }
    writeTimer('drums-interp-time', start);
    write2dArray('drums-interp', interpResults);
  });

  start = performance.now();
  dl.tidy(() => {
    const sample = mvae.sample(5, 32);
    const sampleResults: number[][] = [];
    for (let i = 0; i < sample.shape[0]; i++) {
      const bits: Uint8Array[] = [];
      for (let j = 0; j < sample.shape[1]; j++) {
        const r = dl.slice3d(sample, [i, j, 0], [1, 1, sample.shape[2]]);
        bits.push(r.toInt().dataSync() as Uint8Array);
      }
      sampleResults.push(bitsToInts(bits));
    }
    writeTimer('drums-sample-time', start);
    write2dArray('drums-samples', sampleResults);
  });
  mvae.dispose();
  console.log(dl.memory());
}

async function runDrumsNade(){
  const mvae: MusicVAE = new MusicVAE(DRUMS_NADE_CKPT);
  await mvae.initialize();

  const drums = [
    [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
     0, 0, 0, 1, 0, 4, 0],
    [15, 0, 4, 4, 68, 0, 23, 0, 13, 0, 100, 0, 39, 64, 52, 0, 143, 0, 4, 4, 68,
     0, 23, 0, 13, 0, 100, 0, 7, 0, 36, 0],
    [6, 0, 4, 0, 4, 0, 23, 0, 5, 0, 4, 0, 22, 0, 13, 0, 5, 0, 4, 0, 22, 0, 5, 0,
     5, 0, 22, 0, 20, 0, 21, 0],
    [0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0]];

  const drumsInput: [number[][], number[][], number[][], number[][]] =
    [intsToBits(drums[0], 10), intsToBits(drums[1], 10),
     intsToBits(drums[2], 10), intsToBits(drums[3], 10)];

  write2dArray('nade-inputs', drums);

  let start = performance.now();
  dl.tidy(() => {
    const interp = mvae.interpolate(dl.tensor3d(drumsInput), 3);
    const interpResults: number[][] = [];
    for (let i = 0; i < interp.shape[0]; i++) {
      const bits: Uint8Array[] = [];
      for (let j = 0; j < interp.shape[1]; j++) {
        const r = interp.slice([i, j, 0], [1, 1, interp.shape[2]]);
        bits.push(r.toBool().dataSync() as Uint8Array);
      }
      interpResults.push(bitsToInts(bits));
    }
    writeTimer('nade-interp-time', start);
    write2dArray('nade-interp', interpResults);
  });

  start = performance.now();
  dl.tidy(() => {
    const sample = mvae.sample(5, 32);
    const sampleResults: number[][] = [];
    for (let i = 0; i < sample.shape[0]; i++) {
      const bits: Uint8Array[] = [];
      for (let j = 0; j < sample.shape[1]; j++) {
        const r = sample.slice([i, j, 0], [1, 1, sample.shape[2]]);
        bits.push(r.toBool().dataSync() as Uint8Array);
      }
      sampleResults.push(bitsToInts(bits));
    }
    writeTimer('nade-sample-time', start);
    write2dArray('nade-samples', sampleResults);
  });
  mvae.dispose();
  console.log(dl.memory());
}

async function runMel(){
  const mvae:MusicVAE = new MusicVAE(MEL_CKPT);
  await mvae.initialize();

  const teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0,
                  0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0];
  const teaPots: number[][][] =
      [intsToOneHot(teaPot, 90), intsToOneHot(teaPot.slice(0).reverse(), 90)];

  write2dArray('mel-inputs', [teaPot, teaPot.slice(0).reverse()]);

  let start = performance.now();

  dl.tidy(()=> {
    const interp =  dl.tidy(() => mvae.interpolate(dl.tensor3d(teaPots), 5));
    const interpResults: Int32Array[] = [];
    for (let i = 0; i < interp.shape[0]; i++) {
      const r = interp.slice([i, 0, 0], [1, interp.shape[1], 1]);
      interpResults.push(r.toInt().dataSync() as Int32Array);
    }
    document.getElementById('mel-interp').innerHTML = interpResults.map(
        r => r.toString()).join('<br>');
    writeTimer('mel-interp-time', start);
  });

  start = performance.now();
  dl.tidy(()=> {
    const sample = mvae.sample(5, 32);
    const sampleResults: Int32Array[] = [];
    for (let i = 0; i < sample.shape[0]; i++) {
      const r = sample.slice([i, 0, 0], [1, sample.shape[1], 1]);
      sampleResults.push(r.toInt().dataSync() as Int32Array);
    }
    document.getElementById('mel-samples').innerHTML = sampleResults.map(
        r => r.toString()).join('<br>');
    writeTimer('mel-sample-time', start);

  });
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
