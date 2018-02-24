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

  let start = Date.now();

  let interp = await mvae.interpolate(drumsInput, 11);
  console.log('drums CAT - interpolate: ' + (Date.now() - start) / 1000.);
  for (let i = 0; i < interp.shape[0]; i++) {
    let bits: Int32Array[] = [];
    for (let j = 0; j < interp.shape[1]; j++) {
      const r = dl.slice3d(interp, [i, j, 0], [1, 1, interp.shape[2]]);
      bits.push(r.asType("int32").dataSync());
    }
    console.log(bitsToInts(bits));
  }
  console.log('drums CAT interp - gpu format data: ' + (Date.now() - start) / 1000.);

  start = Date.now();
  let sample = await mvae.sample(10, 32);
  console.log('drums CAT - sample: ' + (Date.now() - start) / 1000.);
  for (let i = 0; i < sample.shape[0]; i++) {
    let bits: Int32Array[] = [];
    for (let j = 0; j < interp.shape[1]; j++) {
      const r = dl.slice3d(sample, [i, j, 0], [1, 1, sample.shape[2]])
      bits.push(r.asType("int32").dataSync());
    }
    console.log(bitsToInts(bits));
  }
  console.log('drums CAT sample - gpu format data: ' + (Date.now() - start) / 1000.);
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

  let start = Date.now();

  let interp = await mvae.interpolate(drumsInput, 11);
  console.log('drums NADE - interpolate: ' + (Date.now() - start) / 1000.);
  for (let i = 0; i < interp.shape[0]; i++) {
    let bits: Int32Array[] = [];
    for (let j = 0; j < interp.shape[1]; j++) {
      const r = dl.slice3d(interp, [i, j, 0], [1, 1, interp.shape[2]]);
      bits.push(r.asType("int32").dataSync());
    }
    console.log(bitsToInts(bits));
  }
  console.log('drums NADE interp - gpu format data: ' + (Date.now() - start) / 1000.);

  start = Date.now();
  let sample = await mvae.sample(10, 32);
  console.log('drum NADE - sample: ' + (Date.now() - start) / 1000.);
  for (let i = 0; i < sample.shape[0]; i++) {
    let bits: Int32Array[] = [];
    for (let j = 0; j < interp.shape[1]; j++) {
      const r = dl.slice3d(sample, [i, j, 0], [1, 1, sample.shape[2]])
      bits.push(r.asType("int32").dataSync());
    }
    console.log(bitsToInts(bits));
  }
  console.log('drums NADE sample - gpu format data: ' + (Date.now() - start) / 1000.);
}

async function initializedMel(){

  const mvae:MusicVAE = await new MusicVAE('https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/mel_small').initialize();

  const teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0, 0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0];
  const teaPots: [number[][], number[][]] = [
      intsToOneHot(teaPot, 90), intsToOneHot(teaPot.slice(0).reverse(), 90)];

  let start = Date.now();

  let data = await mvae.interpolate(teaPots, 11);
  console.log('mel - interpolate: ' + (Date.now() - start) / 1000.);
  for (let i = 0; i < data.shape[0]; i++) {
    const r = math.slice3D(data, [i, 0, 0], [1, data.shape[1], 1]);
    console.log(r.dataSync());
  }
  console.log('mel - gpu format data: ' + (Date.now() - start) / 1000.);
}

try {
	initializeDrums();
    initializedDrumsNade();
    initializedMel();
} catch (err){
	console.error(err);
}
