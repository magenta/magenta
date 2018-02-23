/* Copyright 2017 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tslint:disable-next-line:max-line-length
import { Array1D, Array2D, Array3D, CheckpointLoader, ENV, MatrixOrientation, NDArray, Scalar } from 'deeplearn';

const DECODER_CELL_FORMAT = "decoder/multi_rnn_cell/cell_%d/lstm_cell/"

let console: any = window.console;

const DEBUG = true;
if (!DEBUG) {
  console = {};
  console.log = () => { };
}
const forgetBias = Scalar.new(1.0);
const math = ENV.math;

class LayerVars {
  kernel: Array2D;
  bias: Array1D;
  constructor(kernel: Array2D, bias: Array1D) {
    this.kernel = kernel;
    this.bias = bias;
  }
}

function dense(vars: LayerVars, inputs: Array2D) {
  const weightedResult = math.matMul(inputs, vars.kernel);
  return math.add(weightedResult, vars.bias) as Array2D;
}

export class Nade {
  encWeights: Array2D;
  decWeightsT: Array2D;
  numDims: number;
  numHidden: number;

  constructor(encWeights: Array3D, decWeightsT: Array3D) {
    this.numDims = encWeights.shape[0];
    this.numHidden = encWeights.shape[2];

    this.encWeights = encWeights.as2D(this.numDims, this.numHidden);
    this.decWeightsT = decWeightsT.as2D(this.numDims, this.numHidden);
  }

  sample(encBias: Array2D, decBias: Array2D) {
    const batchSize = encBias.shape[0];

    let samples: Array2D;
    let a = math.clone(encBias);

    for (let i = 0; i < this.numDims; i++) {
      let h = math.sigmoid(a);
      let encWeights_i = math.slice2D(
        this.encWeights, [i, 0], [1, this.numHidden]);
      let decWeightsT_i = math.slice2D(
        this.decWeightsT, [i, 0], [1, this.numHidden]);
      let decBias_i = math.slice2D(decBias, [0, i], [batchSize, 1])
      let condLogits_i = math.add(
        decBias_i,
        math.matMul(
          h, decWeightsT_i, MatrixOrientation.REGULAR,
          MatrixOrientation.TRANSPOSED));
      let condProbs_i = math.sigmoid(condLogits_i);

      let samples_i = math.equal(
        math.clip(condProbs_i, 0, 0.5), Scalar.new(0.5)) as Array2D;  // >= 0.5
      if (i < this.numDims - 1) {
        a = math.add(a, math.matMul(samples_i, encWeights_i)) as Array2D;
      }

      samples = (
        i ? math.concat2D(samples, samples_i, 1) :
          samples_i.as2D(batchSize, 1));
    }
    return samples;
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

  private runLstm(inputs: Array3D, lstmVars: LayerVars, reverse: boolean) {
    const batchSize = inputs.shape[0];
    const length = inputs.shape[1];
    const outputSize = inputs.shape[2];
    let state: Array2D[] = [
      math.track(Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4])),
      math.track(Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4]))
    ]
    let lstm = math.basicLSTMCell.bind(math, forgetBias, lstmVars.kernel, lstmVars.bias);
    for (let i = 0; i < length; i++) {
      let index = reverse ? length - 1 - i : i;
      state = lstm(
        math.slice3D(inputs, [0, index, 0], [batchSize, 1, outputSize]).as2D(batchSize, outputSize),
        state[0], state[1]);
    }
    return state;
  }

  encode(sequence: Array3D): Array2D {
    const fwState = this.runLstm(sequence, this.lstmFwVars, false);
    const bwState = this.runLstm(sequence, this.lstmBwVars, true);
    const finalState = math.concat2D(fwState[1], bwState[1], 1)
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

  decode(z: Array2D, length: number) {
    const batchSize = z.shape[0];

    // Initialize LSTMCells.
    const lstmCells = []
    let c: Array2D[] = [];
    let h: Array2D[] = [];
    const initialStates = math.tanh(dense(this.zToInitStateVars, z));
    let stateOffset = 0;
    for (let i = 0; i < this.lstmCellVars.length; ++i) {
      const lv = this.lstmCellVars[i];
      const stateWidth = lv.bias.shape[0] / 4;
      lstmCells.push(math.basicLSTMCell.bind(math, forgetBias, lv.kernel, lv.bias))
      c.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
      h.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
    }

    // Generate samples.
    let samples: Array3D;
    let nextInput = math.track(Array2D.zeros([batchSize, this.outputDims]));
    for (let i = 0; i < length; ++i) {
      let output = math.multiRNNCell(lstmCells, math.concat2D(nextInput, z, 1), c, h);
      c = output[0];
      h = output[1];
      const logits = dense(this.outputProjectVars, h[h.length - 1]);

      let timeSamples: Array3D;
      if (this.nade == null) {
        let timeLabels = math.argMax(logits, 1).as1D();
        nextInput = math.oneHot(timeLabels, this.outputDims);
        timeSamples = timeLabels.as3D(batchSize, 1, 1);
      } else {
        let encBias = math.slice2D(
          logits, [0, 0], [batchSize, this.nade.numHidden]);
        let decBias = math.slice2D(
          logits, [0, this.nade.numHidden], [batchSize, this.nade.numDims]);
        nextInput = this.nade.sample(encBias, decBias);
        timeSamples = nextInput.as3D(batchSize, 1, this.outputDims);
      }
      samples = i ? math.concat3D(samples, timeSamples, 1) : timeSamples;
    }
    return samples;
  }
}


export const isDeviceSupported:boolean = isWebGLSupported() && !isSafari();


function initialize(checkpointURL: string) {
	const reader = new CheckpointLoader(checkpointURL);
	return reader.getAllVariables().then(
		(vars: { [varName: string]: NDArray }) => {
			const encLstmFw = new LayerVars(
				vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as Array2D,
				vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias'] as Array1D);
			const encLstmBw = new LayerVars(
				vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as Array2D,
				vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias'] as Array1D);
			const encMu = new LayerVars(
				vars['encoder/mu/kernel'] as Array2D,
				vars['encoder/mu/bias'] as Array1D);
			const encPresig = new LayerVars(
				vars['encoder/sigma/kernel'] as Array2D,
				vars['encoder/sigma/bias'] as Array1D);

			let decLstmLayers: LayerVars[] = [];
			let l = 0;
			while (true) {
				const cell_prefix = DECODER_CELL_FORMAT.replace('%d', l.toString());
				if (!(cell_prefix + 'kernel' in vars)) {
					break;
				}
				decLstmLayers.push(new LayerVars(
					vars[cell_prefix + 'kernel'] as Array2D,
					vars[cell_prefix + 'bias'] as Array1D));
				++l;
			}

			const decZtoInitState = new LayerVars(
				vars['decoder/z_to_initial_state/kernel'] as Array2D,
				vars['decoder/z_to_initial_state/bias'] as Array1D);
			const decOutputProjection = new LayerVars(
				vars['decoder/output_projection/kernel'] as Array2D,
				vars['decoder/output_projection/bias'] as Array1D);
			let nade = (('decoder/nade/w_enc' in vars) ?
				new Nade(
					vars['decoder/nade/w_enc'] as Array3D,
					vars['decoder/nade/w_dec_t'] as Array3D) : null);
			return [
				new Encoder(encLstmFw, encLstmBw, encMu, encPresig),
				new Decoder(decLstmLayers, decZtoInitState, decOutputProjection, nade)
			];
		})
}

class MusicVAE {

	checkpointURL:string;
  encoder: Encoder;
  decoder: Decoder;

  constructor(checkpointURL:string) {
		this.checkpointURL = checkpointURL;
	}

	async initialize() {
		return initialize(this.checkpointURL)
			.then((encoder_decoder:[Encoder, Decoder])=>{
				this.encoder = encoder_decoder[0];
				this.decoder = encoder_decoder[1];
				return this;
			});
	}

	isInitialized() {
		return (!!this.encoder && !!this.decoder);
	}


  async interpolate(noteSequences: number[][][], numSteps: number) {
    if (noteSequences.length != 2 && noteSequences.length != 4) {
      throw new Error('invalid number of note sequences. Requires length 2, or 4');
    }

    const z = math.scope((keep, track) => {
      const startSeq = track(Array2D.new(
        [noteSequences[0].length, noteSequences[0][0].length], noteSequences[0]));
      const startSeq3D = startSeq.as3D(1, startSeq.shape[0], startSeq.shape[1]);

      let batchedInput: Array3D = startSeq3D;
      for (let i = 1; i < noteSequences.length; i++) {
        const endSeq = track(Array2D.new(
          [noteSequences[i].length, noteSequences[i][0].length], noteSequences[i]));
        const endSeq3D = endSeq.as3D(1, endSeq.shape[0], endSeq.shape[1]);
        batchedInput = math.concat3D(batchedInput, endSeq3D, 0);
      }
      // Compute z values.
      return this.encoder.encode(batchedInput);
    });

    // Interpolate.
    const range: number[] = [];
    for (let i = 0; i < numSteps; i++) {
      range.push(i / (numSteps - 1));
    }

    const interpolatedZs = await math.scope(async (keep, track) => {
      const rangeArray: Array1D = track(Array1D.new(range));

      const z0 = math.slice2D(z, [0, 0], [1, z.shape[1]]).as1D();
      const z1 = math.slice2D(z, [1, 0], [1, z.shape[1]]).as1D();

      if (noteSequences.length == 2) {
        const zDiff = math.subtract(z1, z0) as Array1D;
        const diffRange = math.outerProduct(rangeArray, zDiff) as Array2D;
        return math.add(diffRange, z0) as Array2D;
      } else if (noteSequences.length == 4) {
        const z2 = math.slice2D(z, [2, 0], [1, z.shape[1]]).as1D();
        const z3 = math.slice2D(z, [3, 0], [1, z.shape[1]]).as1D();

        const one = Scalar.new(1.0).as1D();
        const revRangeArray = math.subtract(one, rangeArray) as Array1D;

        const r = range.length;
        let finalZs = math.multiply(z0, math.outerProduct(revRangeArray, revRangeArray).as3D(r, r, 1)) as Array3D;  // brodcasting
        finalZs = math.addStrict(finalZs, math.multiply(z1, math.outerProduct(rangeArray, revRangeArray).as3D(r, r, 1))) as Array3D;
        finalZs = math.addStrict(finalZs, math.multiply(z2, math.outerProduct(revRangeArray, rangeArray).as3D(r, r, 1))) as Array3D;
        finalZs = math.addStrict(finalZs, math.multiply(z3, math.outerProduct(rangeArray, rangeArray).as3D(r, r, 1))) as Array3D;
        return finalZs.as2D(range.length * range.length, z.shape[1]);
      } else {
        throw new Error('invalid number of note sequences. Requires length 2, or 4');
      }
    });

    return math.scope(() => {
      return this.decoder.decode(interpolatedZs, noteSequences[0].length);
    });
  }

  async sample(numSamples: number, numSteps: number) {
    return math.scope((track, keep) => {
      const randZs = track(Array2D.randNormal([numSamples, this.decoder.zDims]));
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

function bitsToInts(bits: Int32Array[]) {
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
////////////
// pulled from deeplearnjs/demos/utils.ts
// ideally could be retrieved from NPM modules or internally from deeplearn via NPM
function isWebGLSupported(): boolean {
  return ENV.get('WEBGL_VERSION') >= 1;
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
