import { MusicVAE, intsToBits, bitsToInts, intsToOneHot } from '../index';
import * as dl from 'deeplearn'

async function initializeDrums(){

  const mvae:MusicVAE = await new MusicVAE('https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/drums_small_hikl').initialize();

  const drums = [
    [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 4, 0],
    [15, 0, 4, 4, 68, 0, 23, 0, 13, 0, 100, 0, 39, 64, 52, 0, 143, 0, 4, 4, 68, 0, 23, 0, 13, 0, 100, 0, 7, 0, 36, 0],
    [6, 0, 4, 0, 4, 0, 23, 0, 5, 0, 4, 0, 22, 0, 13, 0, 5, 0, 4, 0, 22, 0, 5, 0, 5, 0, 22, 0, 20, 0, 21, 0],
    [0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];

  const drumsInput: [number[][], number[][], number[][], number[][]] =
    [intsToBits(drums[0], 10), intsToBits(drums[1], 10), intsToBits(drums[2], 10), intsToBits(drums[3], 10)];

  document.getElementById('drums-inputs').innerHTML = drums.map(d => d.toString()).join('<br>');

  let start = Date.now();

  let interp = await mvae.interpolate(drumsInput, 3);
  document.getElementById('drums-interp-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  let interpResults: number[][] = [];
  for (let i = 0; i < interp.shape[0]; i++) {
    let bits: Uint8Array[] = [];
    for (let j = 0; j < interp.shape[1]; j++) {
      const r: dl.Array3D  = dl.slice3d(interp, [i, j, 0], [1, 1, interp.shape[2]]);
      bits.push(r.toInt().dataSync() as Uint8Array);
    }
    interpResults.push(bitsToInts(bits));
  }
  document.getElementById('drums-interp-format-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  document.getElementById('drums-interp').innerHTML = interpResults.map(r => r.toString()).join('<br>');

  start = Date.now();
  let sample = await mvae.sample(5, 32);
  document.getElementById('drums-sample-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  let sampleResults: number[][] = [];
  for (let i = 0; i < sample.shape[0]; i++) {
    let bits: Uint8Array[] = [];
    for (let j = 0; j < interp.shape[1]; j++) {
      const r = dl.slice3d(sample, [i, j, 0], [1, 1, sample.shape[2]])
      bits.push(r.toInt().dataSync() as Uint8Array);
    }
    sampleResults.push(bitsToInts(bits));
  }
  document.getElementById('drums-sample-format-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  document.getElementById('drums-samples').innerHTML = sampleResults.map(r => r.toString()).join('<br>');
}


async function initializedDrumsNade(){

  const mvae:MusicVAE = await new MusicVAE('https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/drums_nade_9').initialize();

  const drums = [
    [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 4, 0],
    [15, 0, 4, 4, 68, 0, 23, 0, 13, 0, 100, 0, 39, 64, 52, 0, 143, 0, 4, 4, 68, 0, 23, 0, 13, 0, 100, 0, 7, 0, 36, 0],
    [6, 0, 4, 0, 4, 0, 23, 0, 5, 0, 4, 0, 22, 0, 13, 0, 5, 0, 4, 0, 22, 0, 5, 0, 5, 0, 22, 0, 20, 0, 21, 0],
    [0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];

  const drumsInput: [number[][], number[][], number[][], number[][]] =
    [intsToBits(drums[0], 10), intsToBits(drums[1], 10), intsToBits(drums[2], 10), intsToBits(drums[3], 10)];

  document.getElementById('nade-inputs').innerHTML = drums.map(d => d.toString()).join('<br>');

  let start = Date.now();

  let interp = await mvae.interpolate(drumsInput, 3);
  document.getElementById('nade-interp-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  let interpResults: number[][] = [];
  for (let i = 0; i < interp.shape[0]; i++) {
    let bits: Uint8Array[] = [];
    for (let j = 0; j < interp.shape[1]; j++) {
      const r = dl.slice3d(interp, [i, j, 0], [1, 1, interp.shape[2]]);
      bits.push(r.toBool().dataSync() as Uint8Array);
    }
    interpResults.push(bitsToInts(bits));
  }
  document.getElementById('nade-interp-format-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  document.getElementById('nade-interp').innerHTML = interpResults.map(r => r.toString()).join('<br>');

  start = Date.now();
  let sample = await mvae.sample(5, 32);
  document.getElementById('nade-sample-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  let sampleResults: number[][] = [];
  for (let i = 0; i < sample.shape[0]; i++) {
    let bits: Uint8Array[] = [];
    for (let j = 0; j < sample.shape[1]; j++) {
      const r = dl.slice3d(sample, [i, j, 0], [1, 1, sample.shape[2]])
      bits.push(r.toBool().dataSync() as Uint8Array);
    }
    sampleResults.push(bitsToInts(bits));
  }
  document.getElementById('nade-sample-format-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  document.getElementById('nade-samples').innerHTML = sampleResults.map(r => r.toString()).join('<br>');
}

async function initializedMel(){

  const mvae:MusicVAE = await new MusicVAE('https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/mel_small').initialize();

  const teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0, 0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0];
  const teaPots: number[][][] =
      [intsToOneHot(teaPot, 90), intsToOneHot(teaPot.slice(0).reverse(), 90)];

  document.getElementById('mel-inputs').innerHTML = [teaPot, teaPot.slice(0).reverse()].map(r => r.toString()).join('<br>');

  let start = Date.now();

  let interp = await mvae.interpolate(teaPots, 5);
  document.getElementById('mel-interp-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  let interpResults: Int32Array[] = [];
  for (let i = 0; i < interp.shape[0]; i++) {
    const r = dl.slice3d(interp, [i, 0, 0], [1, interp.shape[1], 1]);
    interpResults.push(r.toInt().dataSync() as Int32Array);
  }
  document.getElementById('mel-interp').innerHTML = interpResults.map(r => r.toString()).join('<br>');
  document.getElementById('mel-interp-format-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';

  start = Date.now();
  let sample = await mvae.sample(5, 32);
  document.getElementById('mel-sample-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  let sampleResults: Int32Array[] = [];
  for (let i = 0; i < sample.shape[0]; i++) {
    const r = dl.slice3d(sample, [i, 0, 0], [1, sample.shape[1], 1]);
    sampleResults.push(r.toInt().dataSync() as Int32Array);
  }
  document.getElementById('mel-sample-format-time').innerHTML = ((Date.now() - start) / 1000.).toString() + 's';
  document.getElementById('mel-samples').innerHTML = sampleResults.map(r => r.toString()).join('<br>');
}

try {
  initializeDrums();
  initializedDrumsNade();
  initializedMel();
} catch (err){
  console.error(err);
}
