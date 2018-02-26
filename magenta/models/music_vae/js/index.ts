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
import * as dl from 'deeplearn'

const DECODER_CELL_FORMAT = "decoder/multi_rnn_cell/cell_%d/lstm_cell/"

const forgetBias = dl.scalar(1.0);

class LayerVars {
  kernel: dl.Tensor2D;
  bias: dl.Tensor1D;
  constructor(kernel: dl.Tensor2D, bias: dl.Tensor1D) {
    this.kernel = kernel;
    this.bias = bias;
  }
}

function dense(vars: LayerVars, inputs: dl.Tensor2D) {
  return inputs.matMul(vars.kernel).add(vars.bias) as dl.Tensor2D;
  // const weightedResult = math.matMul(inputs, vars.kernel);
  // return math.add(weightedResult, vars.bias) as Array2D;
}

export class Nade {
  encWeights: dl.Tensor2D;
  decWeightsT: dl.Tensor2D;
  numDims: number;
  numHidden: number;

  constructor(encWeights: dl.Tensor3D, decWeightsT: dl.Tensor3D) {
    this.numDims = encWeights.shape[0];
    this.numHidden = encWeights.shape[2];

    this.encWeights = encWeights.as2D(this.numDims, this.numHidden);
    this.decWeightsT = decWeightsT.as2D(this.numDims, this.numHidden);
  }

  sample(encBias: dl.Tensor2D, decBias: dl.Tensor2D) {
    const batchSize = encBias.shape[0];

    let samples: dl.Tensor1D[] = [];
    let a = dl.clone(encBias);

    for (let i = 0; i < this.numDims; i++) {
      let h = dl.sigmoid(a);
      let encWeights_i = dl.slice2d(
        this.encWeights, [i, 0], [1, this.numHidden]).as1D();
      let decWeightsT_i = dl.slice2d(
        this.decWeightsT, [i, 0], [1, this.numHidden]);
      let decBias_i = dl.slice2d(decBias, [0, i], [batchSize, 1]);
      let condLogits_i = decBias_i.add(dl.matMul(h, decWeightsT_i, false, true));
      let condProbs_i = dl.sigmoid(condLogits_i);

      let samples_i = dl.greater(condProbs_i, dl.scalar(0.5)).toFloat().as1D();
      if (i < this.numDims - 1) {
        a = a.add(dl.outerProduct(samples_i.toFloat(), encWeights_i)) as dl.Tensor2D;
      }

      samples.push(samples_i);
    }
    return dl.stack(samples, 1) as dl.Tensor2D;
  }
}

class Encoder {
  lstmFwVars: LayerVars;
  lstmBwVars: LayerVars;
  muVars: LayerVars;
  presigVars: LayerVars;
  zDims: number;

  constructor(lstmFwVars: LayerVars, lstmBwVars: LayerVars, muVars: LayerVars, presigVars: LayerVars) {
    this.lstmFwVars = lstmFwVars;
    this.lstmBwVars = lstmBwVars;
    this.muVars = muVars;
    this.presigVars = presigVars;
    this.zDims = this.muVars.bias.shape[0];
  }

  private runLstm(inputs: dl.Tensor3D, lstmVars: LayerVars, reverse: boolean) {
    const batchSize = inputs.shape[0];
    const length = inputs.shape[1];
    const outputSize = inputs.shape[2];
    return dl.tidy(()=> {
      let state: [dl.Tensor2D, dl.Tensor2D] = [
        dl.zeros([batchSize, lstmVars.bias.shape[0] / 4]),
        dl.zeros([batchSize, lstmVars.bias.shape[0] / 4])
      ];
      let lstm = (data: dl.Tensor2D, state: [dl.Tensor2D, dl.Tensor2D]) =>
          dl.basicLSTMCell(forgetBias, lstmVars.kernel, lstmVars.bias, data, state[0], state[1]);
      for (let i = 0; i < length; i++) {
        let index = reverse ? length - 1 - i : i;
        state = lstm(
          dl.slice3d(inputs, [0, index, 0], [batchSize, 1, outputSize]).as2D(batchSize, outputSize),
          state);
      }
      return state;
    });
  }

  encode(sequence: dl.Tensor3D): dl.Tensor2D {
    const fwState = this.runLstm(sequence, this.lstmFwVars, false);
    const bwState = this.runLstm(sequence, this.lstmBwVars, true);
    // should this be dl.concat2d([fwState.get(1), bwState[], 1)?;
    const finalState = dl.concat2d([fwState[1], bwState[1]], 1);
    const mu = dense(this.muVars, finalState);
    return mu;
  }
}

class Decoder {
  lstmCellVars: LayerVars[];
  zToInitStateVars: LayerVars;
  outputProjectVars: LayerVars;
  zDims: number;
  outputDims: number;
  nade: Nade;

  constructor(
      lstmCellVars: LayerVars[], zToInitStateVars: LayerVars,
      outputProjectVars: LayerVars, nade: Nade) {
    this.lstmCellVars = lstmCellVars;
    this.zToInitStateVars = zToInitStateVars;
    this.outputProjectVars = outputProjectVars;
    this.zDims = this.zToInitStateVars.kernel.shape[0];
    this.outputDims = (
      (nade != null) ? nade.numDims : outputProjectVars.bias.shape[0]);
    this.nade = nade;
  }

  decode(z: dl.Tensor2D, length: number) {
    const batchSize = z.shape[0];

    // Initialize LSTMCells.
    let lstmCells : dl.LSTMCell[] = [];
    let c: dl.Tensor2D[] = [];
    let h: dl.Tensor2D[] = [];
    const initialStates = dl.tanh(dense(this.zToInitStateVars, z));
    let stateOffset = 0;
    for (let i = 0; i < this.lstmCellVars.length; ++i) {
      const lv = this.lstmCellVars[i];
      const stateWidth = lv.bias.shape[0] / 4;
      lstmCells.push(
       (data: dl.Tensor2D, c: dl.Tensor2D, h: dl.Tensor2D) =>
            dl.basicLSTMCell(forgetBias, lv.kernel, lv.bias, data, c, h));
      c.push(dl.slice2d(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
      h.push(dl.slice2d(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
    }

    // Generate samples.
    return dl.tidy(() => {
      let samples: dl.Tensor2D[] = [];
      let nextInput = dl.zeros([batchSize, this.outputDims]) as dl.Tensor2D;
      for (let i = 0; i < length; ++i) {
        let output = dl.multiRNNCell(
            lstmCells, dl.concat2d([nextInput, z], 1), c, h);
        c = output[0];
        h = output[1];
        const logits = dense(this.outputProjectVars, h[h.length - 1]);

        let timeSamples: dl.Tensor2D;
        if (this.nade == null) {
          let timeLabels = dl.argMax(logits, 1).as1D();
          nextInput = dl.oneHot(timeLabels, this.outputDims).toFloat();
          timeSamples = timeLabels.as2D(batchSize, 1);
        } else {
          let encBias = dl.slice2d(
            logits, [0, 0], [batchSize, this.nade.numHidden]);
          let decBias = dl.slice2d(
            logits, [0, this.nade.numHidden], [batchSize, this.nade.numDims]);
          nextInput = this.nade.sample(encBias, decBias);
          timeSamples = nextInput;
        }
        samples.push(timeSamples);
      }

      return dl.stack(samples, 1) as dl.Tensor3D;
    });
  }

}

class MusicVAE {
  checkpointURL:string;
  encoder: Encoder;
  decoder: Decoder;

  constructor(checkpointURL:string) {
    this.checkpointURL = checkpointURL;
  }

	async initialize() {
    const reader = new dl.CheckpointLoader(this.checkpointURL);
    return reader.getAllVariables().then(
        (vars: { [varName: string]: dl.Tensor }) => {
            const encLstmFw = new LayerVars(
                vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as dl.Tensor2D,
                vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias'] as dl.Tensor1D);
            const encLstmBw = new LayerVars(
                vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as dl.Tensor2D,
                vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias'] as dl.Tensor1D);
            const encMu = new LayerVars(
                vars['encoder/mu/kernel'] as dl.Tensor2D,
                vars['encoder/mu/bias'] as dl.Tensor1D);
            const encPresig = new LayerVars(
                vars['encoder/sigma/kernel'] as dl.Tensor2D,
                vars['encoder/sigma/bias'] as dl.Tensor1D);

            let decLstmLayers: LayerVars[] = [];
            let l = 0;
            while (true) {
                const cell_prefix = DECODER_CELL_FORMAT.replace('%d', l.toString());
                if (!(cell_prefix + 'kernel' in vars)) {
                    break;
                }
                decLstmLayers.push(new LayerVars(
                    vars[cell_prefix + 'kernel'] as dl.Tensor2D,
                    vars[cell_prefix + 'bias'] as dl.Tensor1D));
                ++l;
            }

            const decZtoInitState = new LayerVars(
                vars['decoder/z_to_initial_state/kernel'] as dl.Tensor2D,
                vars['decoder/z_to_initial_state/bias'] as dl.Tensor1D);
            const decOutputProjection = new LayerVars(
                vars['decoder/output_projection/kernel'] as dl.Tensor2D,
                vars['decoder/output_projection/bias'] as dl.Tensor1D);
            let nade = (('decoder/nade/w_enc' in vars) ?
                new Nade(
                    vars['decoder/nade/w_enc'] as dl.Tensor3D,
                    vars['decoder/nade/w_dec_t'] as dl.Tensor3D) : null);
            return [
                new Encoder(encLstmFw, encLstmBw, encMu, encPresig),
                new Decoder(decLstmLayers, decZtoInitState, decOutputProjection, nade)
            ];
        }).then((encoder_decoder:[Encoder, Decoder])=>{
          this.encoder = encoder_decoder[0];
          this.decoder = encoder_decoder[1];
          return this;
        });
	}

	isInitialized() {
	  return (!!this.encoder && !!this.decoder);
	}

  async interpolate(sequences: dl.Tensor3D, numSteps: number) {
    if (sequences.shape[0] != 2 && sequences.shape[0] != 4) {
      throw new Error('Invalid number of input sequences. Requires length 2, or 4');
    }

    const z = dl.tidy(() => {
      return this.encoder.encode(sequences);
    });

    const interpolatedZs: dl.Tensor2D = await dl.tidy(() => {
      const rangeArray = dl.linspace(0.0, 1.0, numSteps);

      const z0 = dl.slice2d(z, [0, 0], [1, z.shape[1]]).as1D();
      const z1 = dl.slice2d(z, [1, 0], [1, z.shape[1]]).as1D();

      if (sequences.shape[0] == 2) {
        const zDiff = z1.sub(z0) as dl.Tensor1D;
        return dl.outerProduct(rangeArray, zDiff).add(z0) as dl.Tensor2D;
      } else if (sequences.shape[0] == 4) {
        const z2 = dl.slice2d(z, [2, 0], [1, z.shape[1]]).as1D();
        const z3 = dl.slice2d(z, [3, 0], [1, z.shape[1]]).as1D();

        const revRangeArray = dl.scalar(1.0).sub(rangeArray) as dl.Tensor1D;

        const r = numSteps;
        let finalZs = z0.mul(dl.outerProduct(revRangeArray, revRangeArray).as3D(r, r, 1)) as dl.Tensor3D;
        finalZs = dl.addStrict(finalZs, z1.mul(dl.outerProduct(rangeArray, revRangeArray).as3D(r, r, 1))) as dl.Tensor3D;
        finalZs = dl.addStrict(finalZs, z2.mul(dl.outerProduct(revRangeArray, rangeArray).as3D(r, r, 1))) as dl.Tensor3D;
        finalZs = dl.addStrict(finalZs, z3.mul(dl.outerProduct(rangeArray, rangeArray).as3D(r, r, 1))) as dl.Tensor3D;

        return finalZs.as2D(r * r, z.shape[1]);
      } else {
        throw new Error('invalid number of note sequences. Requires length 2, or 4');
      }
    });

    return dl.tidy(() => {
      return this.decoder.decode(interpolatedZs, sequences.shape[1]);
    });
  }

  async sample(numSamples: number, numSteps: number) {
    return dl.tidy(() => {
      const randZs = dl.randomNormal([numSamples, this.decoder.zDims])  as dl.Tensor2D;
      return this.decoder.decode(randZs, numSteps);
    });
  }
}

function intsToBits(ints: number[], depth: number) {
  const bits: number[][] = [];
  for (let i = 0; i < ints.length; i++) {
    const b: number[] = [];
    for (let d = 0; d < depth; d++) {
      b.push(ints[i] >> d & 1);
    }
    if (ints[i] == 0) {
      b[depth - 1] = 1
    }
    bits.push(b)
  }
  return bits;
}

function bitsToInts(bits: Uint8Array[]) {
  const ints: number[] = [];
  for (let i = 0; i < bits.length; i++) {
    let b = 0;
    for (let d = 0; d < bits[i].length; d++) {
      b += (bits[i][d] << d);
    }
    ints.push(b)
  }
  return ints;
}

function intsToOneHot(ints: number[], depth: number) {
  const oneHot: number[][] = [];
  for (let i = 0; i < ints.length; i++) {
    const oh: number[] = [];
    for (let d = 0; d < depth; d++) {
      oh.push(d == ints[i] ? 1 : 0);
    }
    oneHot.push(oh);
  }
  return oneHot;
}

export {
	LayerVars,
	Encoder,
	Decoder,
	MusicVAE,
	intsToBits,
	bitsToInts,
	intsToOneHot
}

export const isDeviceSupported:boolean = isWebGLSupported() && !isSafari();

////////////
// pulled from deeplearnjs/demos/utils.ts
// ideally could be retrieved from NPM modules or internally from deeplearn via NPM
function isWebGLSupported(): boolean {
  return dl.ENV.get('WEBGL_VERSION') >= 1;
}

function isSafari(): boolean {
  const ua = navigator.userAgent.toLowerCase();
  if (ua.indexOf('safari') !== -1) {
    if (ua.indexOf('chrome') > -1) {
      return false;
    } else {
      return true;
    }
  }
  return false;
}
