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
import { INoteSequence, DataConverter } from './data';

const DECODER_CELL_FORMAT = "decoder/multi_rnn_cell/cell_%d/lstm_cell/";

const forgetBias = dl.scalar(1.0);

/**
 * A class for keeping track of the parameters of an affine transformation.
 *
 * @param kernel A 2-dimensional tensor with the kernel parameters.
 * @param bias A 1-dimensional tensor with the bias parameters.
 */
class LayerVars {
  kernel: dl.Tensor2D;
  bias: dl.Tensor1D;
  constructor(kernel: dl.Tensor2D, bias: dl.Tensor1D) {
    this.kernel = kernel;
    this.bias = bias;
  }
}

/**
 * Helper function to compute an affine transformation.
 *
 * @param vars `LayerVars` containing the `kernel` and `bias` of the
 * transformation.
 * @param inputs A batch of input vectors to transform.
 */
function dense(vars: LayerVars, inputs: dl.Tensor2D) {
  return inputs.matMul(vars.kernel).add(vars.bias) as dl.Tensor2D;
}

/**
 * A Neural Autoregressive Distribution Estimator (NADE).
 */
class Nade {
  encWeights: dl.Tensor2D;
  decWeightsT: dl.Tensor2D;
  numDims: number;
  numHidden: number;

  /**
   * `Nade` contructor.
   *
   * @param encWeights The encoder weights (kernel), sized
   * `[numDims, numHidden, 1]`.
   * @param decWeightsT The transposed decoder weights (kernel), sized
   * `[numDims, numHidden, 1]`.
   */
  constructor(encWeights: dl.Tensor3D, decWeightsT: dl.Tensor3D) {
    this.numDims = encWeights.shape[0];
    this.numHidden = encWeights.shape[2];

    this.encWeights = encWeights.as2D(this.numDims, this.numHidden);
    this.decWeightsT = decWeightsT.as2D(this.numDims, this.numHidden);
  }

  /**
   * Samples from the NADE given a batch of encoder and decoder biases.
   *
   * Selects the MAP (argmax) of each Bernoulli random variable.
   *
   * @param encBias A batch of biases to use when encoding, sized
   * `[batchSize, numHidden]`.
   * @param decBias A batch of biases to use when decoding, sized
   * `[batchSize, numDims]`.
   */
  sample(encBias: dl.Tensor2D, decBias: dl.Tensor2D) {
    const batchSize = encBias.shape[0];
    return dl.tidy(()=> {
      const samples: dl.Tensor1D[] = [];
      let a = encBias.clone();

      for (let i = 0; i < this.numDims; i++) {
        const h = dl.sigmoid(a);
        const encWeightsI = this.encWeights.slice(
            [i, 0], [1, this.numHidden]).as1D();
        const decWeightsTI = this.decWeightsT.slice(
            [i, 0], [1, this.numHidden]);
        const decBiasI = decBias.slice([0, i], [batchSize, 1]);
        const condLogitsI = decBiasI.add(
            dl.matMul(h, decWeightsTI, false, true));
        const condProbsI = condLogitsI.sigmoid();

        const samplesI = condProbsI.greaterEqual(
            dl.scalar(0.5)).toFloat().as1D();
        if (i < this.numDims - 1) {
          a = a.add(
            dl.outerProduct(samplesI.toFloat(), encWeightsI)) as dl.Tensor2D;
        }

        samples.push(samplesI);
      }
     return dl.stack(samples, 1) as dl.Tensor2D;
    });
  }
}

/**
 * Bidirectional LSTM encoder for computing the latent variable `z`.
 */
class Encoder {
  lstmFwVars: LayerVars;
  lstmBwVars: LayerVars;
  muVars: LayerVars;
  zDims: number;

  /**
   * `Encoder` contructor.
   *
   * @param lstmFwVars The forward LSTM `LayerVars`.
   * @param lstmBwVars The backward LSTM `LayerVars`.
   * @param muVars The `LayerVars` for projecting from the final states of the
   * bidirectional LSTM to the mean `mu` of the random variable, `z`.
   */
  constructor(lstmFwVars: LayerVars, lstmBwVars: LayerVars, muVars: LayerVars) {
    this.lstmFwVars = lstmFwVars;
    this.lstmBwVars = lstmBwVars;
    this.muVars = muVars;
    this.zDims = this.muVars.bias.shape[0];
  }

  /**
   * Encodes a batch of input sequences.
   *
   * The final states of the forward and backward LSTMs are concatenated and
   * passed through a dense layer to compute the mean value of the latent
   * variable.
   *
   * @param sequence A batch of sequences to encode, sized
   * `[batchSize, length, depth]`.
   *
   * @returns The means of the latent variables of the encoded sequences, sized
   * `[batchSize, zDims]`.
   */
  encode(sequence: dl.Tensor3D): dl.Tensor2D {
    return dl.tidy(() => {
      const fwState = this.runLstm(sequence, this.lstmFwVars, false);
      const bwState = this.runLstm(sequence, this.lstmBwVars, true);
      const finalState = dl.concat2d([fwState[1], bwState[1]], 1);
      const mu = dense(this.muVars, finalState);
      return mu;
    });
  }

  private runLstm(inputs: dl.Tensor3D, lstmVars: LayerVars, reverse: boolean) {
    const batchSize = inputs.shape[0];
    const length = inputs.shape[1];
    const outputSize = inputs.shape[2];

    let state: [dl.Tensor2D, dl.Tensor2D] = [
      dl.zeros([batchSize, lstmVars.bias.shape[0] / 4]),
      dl.zeros([batchSize, lstmVars.bias.shape[0] / 4])
    ];
    const lstm = (data: dl.Tensor2D, state: [dl.Tensor2D, dl.Tensor2D]) =>
        dl.basicLSTMCell(
          forgetBias, lstmVars.kernel, lstmVars.bias, data, state[0], state[1]);
    for (let i = 0; i < length; i++) {
      const index = reverse ? length - 1 - i : i;
      state = lstm(
          inputs.slice([0, index, 0], [batchSize, 1, outputSize]).as2D(
              batchSize, outputSize),
          state);
    }
    return state;
  }
}

/**
 * LSTM decoder with optional NADE output.
 */
class Decoder {
  lstmCellVars: LayerVars[];
  zToInitStateVars: LayerVars;
  outputProjectVars: LayerVars;
  zDims: number;
  outputDims: number;
  nade: Nade;

  /**
   * `Decoder` contructor.
   *
   * @param lstmCellVars The `LayerVars` for each layer of the decoder LSTM.
   * @param zToInitStateVars The `LayerVars` for projecting from the latent
   * variable `z` to the initial states of the LSTM layers.
   * @param outputProjectVars The `LayerVars` for projecting from the output
   * of the LSTM to the logits of the output categorical distrubtion
   * (if `nade` is null) or to bias values to use in the NADE (if `nade` is
   * not null).
   * @param nade (optional) A `Nade` to use for computing the output vectors at
   * each step. If not given, the final projection values are used as logits
   * for a categorical distrubtion.
   */
  constructor(
      lstmCellVars: LayerVars[], zToInitStateVars: LayerVars,
      outputProjectVars: LayerVars, nade?: Nade) {
    this.lstmCellVars = lstmCellVars;
    this.zToInitStateVars = zToInitStateVars;
    this.outputProjectVars = outputProjectVars;
    this.zDims = this.zToInitStateVars.kernel.shape[0];
    this.outputDims = (nade) ? nade.numDims : outputProjectVars.bias.shape[0];
    this.nade = nade;
  }

  /**
   * Decodes a batch of latent vectors, `z`.
   *
   * If `nade` is parameterized, samples are generated using the MAP (argmax) of
   * the Bernoulli random variables from the NADE, and these bit vector makes up
   * the final dimension of the output.
   *
   * If `nade` is not parameterized, sample labels are generated using the
   * MAP (argmax) of the logits output by the LSTM, and the onehots of those
   * labels makes up the final dimension of the output.
   *
   * @param z A batch of latent vectors to decode, sized `[batchSize, zDims]`.
   * @param length The length of decoded sequences.
   * @param temperature The softmax temperature to use when sampling from the
   * logits. Argmax is used if not provided.
   *
   * @returns A boolean tensor containing the decoded sequences, shaped
   * `[batchSize, length, depth]`.
   */
  decode(z: dl.Tensor2D, length: number, temperature?: number) {
    const batchSize = z.shape[0];

    return dl.tidy(() => {
      // Initialize LSTMCells.
      const lstmCells: dl.LSTMCellFunc[] =  [];
      let c: dl.Tensor2D[] = [];
      let h: dl.Tensor2D[] = [];
      const initialStates = dense(this.zToInitStateVars, z).tanh();
      let stateOffset = 0;
      for (let i = 0; i < this.lstmCellVars.length; ++i) {
        const lv = this.lstmCellVars[i];
        const stateWidth = lv.bias.shape[0] / 4;
        lstmCells.push(
            (data: dl.Tensor2D, c: dl.Tensor2D, h: dl.Tensor2D) =>
            dl.basicLSTMCell(forgetBias, lv.kernel, lv.bias, data, c, h));
        c.push(initialStates.slice([0, stateOffset], [batchSize, stateWidth]));
        stateOffset += stateWidth;
        h.push(initialStates.slice([0, stateOffset], [batchSize, stateWidth]));
        stateOffset += stateWidth;
      }

       // Generate samples.
      const samples: dl.Tensor2D[] = [];
      let nextInput = dl.zeros([batchSize, this.outputDims]) as dl.Tensor2D;
      for (let i = 0; i < length; ++i) {
        const output = dl.multiRNNCell(
            lstmCells, dl.concat2d([nextInput, z], 1), c, h);
        c = output[0];
        h = output[1];
        const logits = dense(this.outputProjectVars, h[h.length - 1]);

        let timeSamples: dl.Tensor2D;
        if (this.nade == null) {
          const timeLabels = (
            temperature ?
            dl.multinomial(logits.div(dl.scalar(temperature)), 1).as1D():
            logits.argMax(1).as1D());
          nextInput = dl.oneHot(timeLabels, this.outputDims).toFloat();
          timeSamples = nextInput.toBool();
        } else {
          const encBias = logits.slice(
              [0, 0], [batchSize, this.nade.numHidden]);
          const decBias = logits.slice(
              [0, this.nade.numHidden], [batchSize, this.nade.numDims]);
          nextInput = this.nade.sample(encBias, decBias);
          timeSamples = nextInput.toBool();
        }
        samples.push(timeSamples);
      }

      return dl.stack(samples, 1) as dl.Tensor3D;
    });
  }
}

/**
 * Main MusicVAE model class.
 *
 * A MusicVAE is a variational autoencoder made up of an `Encoder` and
 * `Decoder`, along with a `DataConverter` for converting between `Tensor`
 * and `NoteSequence` objects for input and output.
 *
 * Exposes methods for interpolation and sampling of musical sequences.
 */
class MusicVAE {
  checkpointURL: string;
  dataConverter: DataConverter;
  encoder: Encoder;
  decoder: Decoder;
  rawVars: {[varName: string]: dl.Tensor};  // Store for disposal.
  /**
   * `MusicVAE` constructor.
   *
   * @param checkpointURL Path to the checkpoint directory.
   * @param dataConverter A `DataConverter` object to use for converting between
   * `NoteSequence` and `Tensor` objects.
   */
  constructor(checkpointURL: string, dataConverter: DataConverter) {
    this.checkpointURL = checkpointURL;
    this.dataConverter = dataConverter;
  }

  /**
   * Disposes of any untracked `Tensors` to avoid GPU memory leaks.
   */
  dispose() {
    Object.keys(this.rawVars).forEach(name => this.rawVars[name].dispose());
    this.encoder = null;
    this.decoder = null;
  }

  /**
   * Loads variables from the checkpoint and instantiates the `Encoder` and
   * `Decoder`.
   */
  async initialize() {
    const reader = new dl.CheckpointLoader(this.checkpointURL);
    const vars = await reader.getAllVariables();

    // Encoder variables.
    // tslint:disable:max-line-length
    const encLstmFw = new LayerVars(
        vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as dl.Tensor2D,
        vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias'] as dl.Tensor1D);
    const encLstmBw = new LayerVars(
        vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as dl.Tensor2D,
        vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias'] as dl.Tensor1D);
    const encMu = new LayerVars(
        vars['encoder/mu/kernel'] as dl.Tensor2D,
        vars['encoder/mu/bias'] as dl.Tensor1D);
    // tslint:enable:max-line-length
    // Decoder LSTM layer variables.
    const decLstmLayers: LayerVars[] = [];
    let l = 0;
    while (true) {
        const cellPrefix = DECODER_CELL_FORMAT.replace('%d', l.toString());
        if (!(cellPrefix + 'kernel' in vars)) {
            break;
        }
        decLstmLayers.push(new LayerVars(
            vars[cellPrefix + 'kernel'] as dl.Tensor2D,
            vars[cellPrefix + 'bias'] as dl.Tensor1D));
        ++l;
    }

    // Other Decoder variables.
    const decZtoInitState = new LayerVars(
        vars['decoder/z_to_initial_state/kernel'] as dl.Tensor2D,
        vars['decoder/z_to_initial_state/bias'] as dl.Tensor1D);
    const decOutputProjection = new LayerVars(
        vars['decoder/output_projection/kernel'] as dl.Tensor2D,
        vars['decoder/output_projection/bias'] as dl.Tensor1D);
    // Optional NADE for the decoder.
    const nade = (('decoder/nade/w_enc' in vars) ?
        new Nade(
            vars['decoder/nade/w_enc'] as dl.Tensor3D,
            vars['decoder/nade/w_dec_t'] as dl.Tensor3D) : null);

    this.encoder = new Encoder(encLstmFw, encLstmBw, encMu);
    this.decoder = new Decoder(
      decLstmLayers, decZtoInitState, decOutputProjection, nade);
    this.rawVars= vars;
    return this;
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
  interpolate(inputSequences: INoteSequence[], numInterps: number) {
    const numSteps = this.dataConverter.numSteps;

    const outputSequences: INoteSequence[] = [];

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

    const outputSequences: INoteSequence[] = [];
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

export {
  LayerVars,
  Encoder,
  Decoder,
  Nade,
  MusicVAE,
};
