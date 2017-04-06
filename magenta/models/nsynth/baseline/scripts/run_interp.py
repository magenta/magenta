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
"""Script for running interp.py."""
import script_utils


def run(v, hp, local=False):
  target_list = [
      "//magenta/models/nsynth/baseline:interp",
  ]
  script_utils.run(target_list, v, hp, local=local, gpu=True)


DATA_DIRS = {
}


#-------------------------------------------------------------------------------
# Recreate ICML Submission
#-------------------------------------------------------------------------------
# Exploring the effect of num_latent on recon and interpolation
def icml_baseline(local=False, run_all=False):
  runs = [64, 256, 512, 1024, 1984] if run_all else [1984]
  for num_latent in runs:
    v = dict(
        job_name="icml_baseline_nlatent{}".format(num_latent),
        config="mag_1_1024nfft",
        interp_name="gold_test",
        data_dir=DATA_DIRS["gold_test"],)
    hp = script_utils.hp_ae
    hp["num_latent"] = num_latent
    run(v, hp, local=local)


if __name__ == "__main__":
  icml_baseline()
