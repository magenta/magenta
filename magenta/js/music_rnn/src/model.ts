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
import * as data from './data';
import { INoteSequence } from '@magenta/core';

// tslint:disable-next-line:max-line-length
const CHECKPOINT_URL = "https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_rnn/dljs/basic_rnn/";

const DEFAULT_MIN_NOTE = 48;
const DEFAULT_MAX_NOTE = 83;

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

  lstmKernel1: dl.Tensor2D;
  lstmBias1: dl.Tensor1D;
  lstmKernel2: dl.Tensor2D;
  lstmBias2: dl.Tensor1D;
  lstmFcB: dl.Tensor1D;
  lstmFcW: dl.Tensor2D;

  initialized: boolean;

  /**
   * `MusicVAE` constructor.
   *
   * @param checkpointURL Path to the checkpoint directory.
   * @param dataConverter A `DataConverter` object to use for converting between
   * `NoteSequence` and `Tensor` objects. If not provided, a `converter.json`
   * file must exist within the checkpoint directory specifying the type and
   * args for the correct `DataConverter`.
   */
  constructor(checkpointURL=CHECKPOINT_URL) {
    this.checkpointURL = checkpointURL;
    this.initialized = false;
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
    // tslint:enable:max-line-length

    this.lstmFcW = vars['fully_connected/weights'] as dl.Tensor2D;
    this.lstmFcB = vars['fully_connected/biases'] as dl.Tensor1D;

    this.initialized = true;
  }

  dispose() {
    this.lstmKernel1.dispose();
    this.lstmBias1.dispose();
    this.lstmKernel2.dispose();
    this.lstmBias2.dispose();
    this.lstmFcW.dispose();
    this.lstmFcB.dispose();
    this.initialized = false;
  }

  async continueSequence(sequence: magenta.INoteSequence, steps: number,
    temperature?: number): Promise<magenta.INoteSequence> {
    magenta.Sequences.assertIsQuantizedSequence(sequence);

    let continuation: INoteSequence;

    if(!this.initialized) {
      await this.initialize();
    }

    dl.tidy(() => {
      const converterIn = new data.MelodyConverter(
        sequence.totalQuantizedSteps, DEFAULT_MIN_NOTE, DEFAULT_MAX_NOTE);
      const inputs = converterIn.toTensor(sequence);

      const forgetBias = dl.scalar(1.0);

      const length:number = inputs.shape[0];
      const outputSize:number = inputs.shape[1];

      let c: dl.Tensor2D[] = [
        dl.zeros([1, this.lstmBias1.shape[0] / 4]),
        dl.zeros([1, this.lstmBias2.shape[0] / 4]),
      ];
      let h: dl.Tensor2D[] = [
        dl.zeros([1, this.lstmBias1.shape[0] / 4]),
        dl.zeros([1, this.lstmBias2.shape[0] / 4]),
      ];

      const lstm1 = (data: dl.Tensor2D, c: dl.Tensor2D, h: dl.Tensor2D) =>
        dl.basicLSTMCell(forgetBias, this.lstmKernel1, this.lstmBias1, data,
          c, h);
      const lstm2 = (data: dl.Tensor2D, c: dl.Tensor2D, h: dl.Tensor2D) =>
        dl.basicLSTMCell(forgetBias, this.lstmKernel2, this.lstmBias2, data,
          c, h);

      // Initialize with input.
      const samples: dl.Tensor1D[] = [];
      for (let i = 0; i < length + steps; i++) {
        let nextInput: dl.Tensor2D;
        if (i < length) {
          nextInput = inputs.slice([
            i, 0], [1, outputSize]).as2D(1, outputSize);
        } else {
          const logits = h[1].matMul(this.lstmFcW).add(this.lstmFcB);
          const sampledOutput = (
            temperature ?
            dl.multinomial(logits.div(dl.scalar(temperature)), 1).as1D():
            logits.argMax(1).as1D());
          nextInput = dl.oneHot(sampledOutput, outputSize).toFloat();
          samples.push(nextInput.as1D());
        }
        const output = dl.multiRNNCell([lstm1, lstm2], nextInput, c, h);
        c = output[0];
        h = output[1];
      }

      const output = dl.stack(samples).as2D(samples.length, outputSize);
      const converterOut = new data.MelodyConverter(
        sequence.totalQuantizedSteps, DEFAULT_MIN_NOTE, DEFAULT_MAX_NOTE);
      continuation = converterOut.toNoteSequence(output);
    });
    return continuation;
  }

}
