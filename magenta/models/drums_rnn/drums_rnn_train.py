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
"""Train and evaluate a drums RNN model."""

# internal imports
import tensorflow as tf

from magenta.models.drums_rnn import drums_rnn_config_flags
from magenta.models.shared import events_rnn_train


def main(unused_argv):
  events_rnn_train.setup_logs()

  config = drums_rnn_config_flags.config_from_flags()
  events_rnn_train.run_from_flags(config)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
