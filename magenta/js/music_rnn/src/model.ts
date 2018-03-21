/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as dl from 'deeplearn';
import * as magenta from '@magenta/core';

const CHECKPOINT_URL = "https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_rnn/dljs/basic_rnn/manifest.json";

const forgetBias = dl.scalar(1.0);

/**
 * Main MusicVAE model class.
 *
 * A MusicVAE is a variational autoencoder made up of an `Encoder` and
 * `Decoder`, along with a `DataConverter` for converting between `Tensor`
 * and `NoteSequence` objects for input and output.
 *
 * Exposes methods for interpolation and sampling of musical sequences.
 */
export class MelodyRnn {
  checkpointURL: string;
  rawVars: {[varName: string]: dl.Tensor};  // Store for disposal.

  lstmKernel1: dl.Tensor2D;
  lstmBias1: dl.Tensor1D;
  lstmKernel2: dl.Tensor2D;
  lstmBias2: dl.Tensor1D;
  lstmFcB: dl.Tensor1D;
  lstmFcW: dl.Tensor2D;

  /**
   * `MusicVAE` constructor.
   *
   * @param checkpointURL Path to the checkpoint directory.
   * @param dataConverter A `DataConverter` object to use for converting between
   * `NoteSequence` and `Tensor` objects. If not provided, a `converter.json`
   * file must exist within the checkpoint directory specifying the type and
   * args for the correct `DataConverter`.
   */
  constructor(checkpointURL: string) {
    this.checkpointURL = checkpointURL;
  }

  /**
   * Disposes of any untracked `Tensors` to avoid GPU memory leaks.
   */
  dispose() {
    Object.keys(this.rawVars).forEach(name => this.rawVars[name].dispose());
  }

  /**
   * Loads variables from the checkpoint and instantiates the `Encoder` and
   * `Decoder`.
   */
  async initialize() {
    const reader = new dl.CheckpointLoader(this.checkpointURL);
    const vars = await reader.getAllVariables();

    // Model variables.
    // tslint:disable:max-line-length
    this.lstmKernel1 = vars['RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix'] as dl.Tensor2D;
    this.lstmBias1 = vars['RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias'] as dl.Tensor1D;
    this.lstmKernel2 = vars['RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix'] as dl.Tensor2D;
    this.lstmBias2 = vars['RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias'] as dl.Tensor1D;

    this.lstmFcB = vars['fully_connected/biases'] as dl.Tensor1D;
    this.lstmFcW = vars['fully_connected/weights'] as dl.Tensor2D;
  }

  primeRnn(sequence: INoteSequence) {
    magenta.Sequences
  }

  /**
   * @returns true iff an `Encoder` and `Decoder` have been instantiated for the
   * model.
   */
  isInitialized() {
    return (!!this.encoder && !!this.decoder);
  }

  /**
   * Interpolates between the input `NoteSequences` in latent space.
   *
   * If 2 sequences are given, a single linear interpolation is computed, with
   * the first output sequence being a reconstruction of sequence A and the
   * final output being a reconstruction of sequence B, with `numInterps`
   * total sequences.
   *
   * If 4 sequences are given, bilinear interpolation is used. The results are
   * returned in row-major order for a matrix with the following layout:
   *   | A . . C |
   *   | . . . . |
   *   | . . . . |
   *   | B . . D |
   * where the letters represent the reconstructions of the four inputs, in
   * alphabetical order, and there are `numInterps` sequences on each
   * edge for a total of `numInterps`^2 sequences.
   *
   * @param inputSequences An array of 2 or 4 `NoteSequences` to interpolate
   * between.
   * @param numInterps The number of pairwise interpolation sequences to
   * return, including the reconstructions. If 4 inputs are given, the total
   * number of sequences will be `numInterps`^2.
   *
   * @returns An array of interpolation `NoteSequence` objects, as described
   * above.
   */
  interpolate(inputSequences: data.INoteSequence[], numInterps: number) {
    const numSteps = this.dataConverter.numSteps;

    const outputSequences: data.INoteSequence[] = [];

    dl.tidy(() => {
      const inputTensors = dl.stack(
        inputSequences.map(
          this.dataConverter.toTensor.bind(this.dataConverter)) as dl.Tensor2D[]
        ) as dl.Tensor3D;

      const outputTensors = this.interpolateTensors(inputTensors, numInterps);
      for (let i = 0; i < outputTensors.shape[0]; ++i) {
        const t = outputTensors.slice(
            [i, 0, 0],
            [1, numSteps, outputTensors.shape[2]]).as2D(
                numSteps, outputTensors.shape[2]);
          outputSequences.push(this.dataConverter.toNoteSequence(t));
      }
    });
    return outputSequences;
  }

  private interpolateTensors(sequences: dl.Tensor3D, numInterps: number) {
    if (sequences.shape[0] !== 2 && sequences.shape[0] !== 4) {
      throw new Error(
          'Invalid number of input sequences. Requires length 2, or 4');
    }

    // Use the mean `mu` of the latent variable as the best estimate of `z`.
    const z =this.encoder.encode(sequences);

    // Compute the interpolations of the latent variable.
    const interpolatedZs: dl.Tensor2D = dl.tidy(() => {
      const rangeArray = dl.linspace(0.0, 1.0, numInterps);

      const z0 = z.slice([0, 0], [1, z.shape[1]]).as1D();
      const z1 = z.slice([1, 0], [1, z.shape[1]]).as1D();

      if (sequences.shape[0] === 2) {
        const zDiff = z1.sub(z0) as dl.Tensor1D;
        return dl.outerProduct(rangeArray, zDiff).add(z0) as dl.Tensor2D;
      } else if (sequences.shape[0] === 4) {
        const z2 = z.slice([2, 0], [1, z.shape[1]]).as1D();
        const z3 = z.slice([3, 0], [1, z.shape[1]]).as1D();

        const revRangeArray = dl.scalar(1.0).sub(rangeArray) as dl.Tensor1D;

        const r = numInterps;
        let finalZs = z0.mul(
            dl.outerProduct(revRangeArray, revRangeArray).as3D(r, r, 1));
        finalZs = dl.addStrict(
            finalZs,
            z1.mul(dl.outerProduct(rangeArray, revRangeArray).as3D(r, r, 1)));
        finalZs = dl.addStrict(
            finalZs,
            z2.mul(dl.outerProduct(revRangeArray, rangeArray).as3D(r, r, 1)));
        finalZs = dl.addStrict(
            finalZs,
            z3.mul(dl.outerProduct(rangeArray, rangeArray).as3D(r, r, 1)));

        return finalZs.as2D(r * r, z.shape[1]);
      } else {
        throw new Error(
          'Invalid number of note sequences. Requires length 2, or 4');
      }
    });

    // Decode the interpolated values of `z`.
    return this.decoder.decode(interpolatedZs, sequences.shape[1]);
  }

  /**
   * Samples sequences from the model prior.
   *
   * @param numSamples The number of samples to return.
   * @param temperature The softmax temperature to use when sampling.
   *
   * @returns An array of sampled `NoteSequence` objects.
   */
  sample(numSamples: number, temperature=0.5) {
    const numSteps = this.dataConverter.numSteps;

    const outputSequences: data.INoteSequence[] = [];
    dl.tidy(() => {
      const outputTensors = this.sampleTensors(
          numSamples, numSteps, temperature);
      for (let i = 0; i < numSamples; ++i) {
        const t = outputTensors.slice(
            [i, 0, 0],
            [1, numSteps, outputTensors.shape[2]]).as2D(
                numSteps, outputTensors.shape[2]);
          outputSequences.push(this.dataConverter.toNoteSequence(t));
      }
    });
    return outputSequences;
  }

  private sampleTensors(
    numSamples: number, numSteps: number, temperature?: number) {
    return dl.tidy(() => {
      const randZs: dl.Tensor2D = dl.randomNormal(
          [numSamples, this.decoder.zDims]);
      return this.decoder.decode(randZs, numSteps, temperature);
    });
  }
}
