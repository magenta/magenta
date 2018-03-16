/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
// Adds quantization support to dl.CheckpointLoader.
import {Tensor} from 'deeplearn';

/**
 * @hidden
 */
export interface CheckpointVariable {
  filename: string;
  shape: number[];
  quantization: {bytes: number, min: number, max:number};
}

/**
 * @hidden
 */
export type CheckpointManifest = {
  [varName: string]: CheckpointVariable
};

const MANIFEST_FILE = 'manifest.json';

export class CheckpointLoader {
  private checkpointManifest: CheckpointManifest;
  private variables: {[varName: string]: Tensor};

  constructor(private urlPath: string) {
    if (this.urlPath.charAt(this.urlPath.length - 1) !== '/') {
      this.urlPath += '/';
    }
  }

  private loadManifest(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', this.urlPath + MANIFEST_FILE);

      xhr.onload = () => {
        this.checkpointManifest = JSON.parse(xhr.responseText);
        resolve();
      };
      xhr.onerror = (error) => {
        throw new Error(
            `${MANIFEST_FILE} not found at ${this.urlPath}. ${error}`);
      };
      xhr.send();
    });
  }

  getCheckpointManifest(): Promise<CheckpointManifest> {
    if (this.checkpointManifest == null) {
      return new Promise<CheckpointManifest>((resolve, reject) => {
        this.loadManifest().then(() => {
          resolve(this.checkpointManifest);
        });
      });
    }
    return new Promise<CheckpointManifest>((resolve, reject) => {
      resolve(this.checkpointManifest);
    });
  }

  getAllVariables(): Promise<{[varName: string]: Tensor}> {
    if (this.variables != null) {
      return new Promise<{[varName: string]: Tensor}>((resolve, reject) => {
        resolve(this.variables);
      });
    }

    return new Promise<{[varName: string]: Tensor}>((resolve, reject) => {
      this.getCheckpointManifest().then(
          (checkpointDefinition: CheckpointManifest) => {
            const variableNames = Object.keys(this.checkpointManifest);

            const variablePromises: Array<Promise<Tensor>> = [];
            for (let i = 0; i < variableNames.length; i++) {
              variablePromises.push(this.getVariable(variableNames[i]));
            }

            Promise.all(variablePromises).then(variables => {
              this.variables = {};
              for (let i = 0; i < variables.length; i++) {
                this.variables[variableNames[i]] = variables[i];
              }
              resolve(this.variables);
            });
          });
    });
  }

  getVariable(varName: string): Promise<Tensor> {
    if (!(varName in this.checkpointManifest)) {
      throw new Error('Cannot load non-existant variable ' + varName);
    }

    const variableRequestPromiseMethod =
        (resolve: (tensor: Tensor) => void, reject: () => void) => {
          const xhr = new XMLHttpRequest();
          xhr.responseType = 'arraybuffer';
          const fname = this.checkpointManifest[varName].filename;
          xhr.open('GET', this.urlPath + fname);

          xhr.onload = () => {
            if (xhr.status === 404) {
              throw new Error(`Not found variable ${varName}`);
            }
            const quantInfo = this.checkpointManifest[varName].quantization;
            let values: Float32Array;
            if (quantInfo) {
              let quantValues: Float32Array;
              if (quantInfo.bytes === 1) {
                quantValues = Float32Array.from(new Uint8Array(xhr.response));
              } else if (quantInfo.bytes === 2) {
                quantValues = Float32Array.from(new Uint16Array(xhr.response));
              } else {
                throw new Error(
                  'Quantization bytes must be either 1 or 2. ' +
                  'Got: ${quantInfo.bytes}');
              }
              let quantConstant = 1.0;
              if (quantInfo.max !== quantInfo.min) {
                quantConstant = (
                    (quantInfo.max - quantInfo.min) /
                    (Math.pow(2, quantInfo.bytes * 8) - 1));
              }
              values = quantValues.map(v => v * quantConstant + quantInfo.min);
            } else {
              values = new Float32Array(xhr.response);
            }
            const tensor =
                Tensor.make(this.checkpointManifest[varName].shape, {values});
            resolve(tensor);
          };
          xhr.onerror = (error) => {
            throw new Error(`Could not fetch variable ${varName}: ${error}`);
          };
          xhr.send();
        };

    if (this.checkpointManifest == null) {
      return new Promise<Tensor>((resolve, reject) => {
        this.loadManifest().then(() => {
          new Promise<Tensor>(variableRequestPromiseMethod).then(resolve);
        });
      });
    }
    return new Promise<Tensor>(variableRequestPromiseMethod);
  }
}
