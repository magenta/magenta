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
"""Utility functions for running borg scripts.

Compiles bazel commands out of flags dict (flags) and hparams dict (hp).
"""

import os


def flag_str(flags_dict):
  """Makes a properly formatted variable string.

  Args:
    flags_dict: A dictionary of flags and their values.

  Returns:
    v_str: A string formatted for the command line.
      "--k0=v0 --k1=v1"
  """
  delimiter = " --"
  v_str = "--"
  v_str += delimiter.join(
      ["{}={}".format(k, v) for k, v in flags_dict.iteritems()])
  v_str = v_str.replace("False", "false").replace("True", "true")
  return v_str


def run(target_list, flags, hp, gpu=True,
        hparams_flag="config_hparams"):
  """Compile and run a job on the commandline.

  Args:
    target_list: A list of paths to all build targets. 
    flags: Dictionary of flag keys and values.
    hp: Dictionary of hyperparmeter keys and values.
    hparams_flag: String, flag name for passing in hparams.

  -------
  Example
  -------
  v = dict(job_name="baseline", worker_replicas=6)
  hp = dict(batch_size=16)
  target_list = [
    "//magenta/models/nsynth/baseline:train",
    "//magenta/models/nsynth/baseline:eval",
  ]
  script_utils.run(target_list, v, hp, gpu=True)
  """

  # Take first entry, remove .par
  targets = os.path.splitext(target_list[0])[0]

  # Format FLAGS and hparams strings
  flags = flag_str(flags, local=local)
  hp = flag_str(hp, local=False)
  print "Flags:", flags
  print "HParams", hp

  cmd = ("bazel run "
         "{targets} -- "
         "{flags} "
         "--{hparams_flag}=\"{hp}\" "
         "--alsologtostderr".format(
             targets=targets,
             flags=flags,
             hparams_flag=hparams_flag,
             hp=hp))
  print "Command", cmd
  os.system(cmd)


#- HParams ---------------------------------------------------------------------
hp_ae = dict(
    num_latent=1984,
    batch_size=8,
    mag_only=True,
    n_fft=1024,
    fw_loss_coeff=10.0,
    fw_loss_cutoff=4000,)

hp_classify = dict(
    n_pitches=128,
    n_qualities=10,
    n_instrument_families=11,
    n_instrument_sources=3,
    batch_size=32,
    mag_only=True,
    n_fft=1024,
    join_family_source=False,)
