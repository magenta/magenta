#!/usr/bin/python2.7
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script for running make_test_set.py."""
import script_utils


def run(v, hp):
  target_list = [
      "//magenta/models/nsynth/baseline:eval"
  ]
  script_utils.run(target_list, v, hp, local=True, gpu=True)


#-------------------------------------------------------------------------------
# Recreate ICML Submission
#-------------------------------------------------------------------------------
# Exploring the effect of num_latent on recon and interpolation
def icml_baseline(run_all=False):
  runs = [64, 256, 512, 1024, 1984] if run_all else [1984]
  for num_latent in runs:
    v = dict(
        dataset="NSYNTH_RC4_EVAL",
        model="ae",
        config="mag_1_1024nfft",
        n_eval=128,)
    hp = script_utils.hp_ae
    hp["num_latent"] = num_latent
    run(v, hp)


if __name__ == "__main__":
  icml_baseline()
