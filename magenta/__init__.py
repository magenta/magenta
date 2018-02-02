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

r"""Pulls in all magenta libraries that are in the public API.

To regenerate this list based on the py_library dependencies of //magenta:

bazel query 'kind(py_library, deps(//magenta))' | \
  grep '//magenta' | \
  egrep  -v "/([^:/]+):\1$" | \
  sed -e 's/\/\//import /' -e 's/\//./' -e 's/:/./' -e  's/py_pb2/pb2/' | \
  LANG=C sort
"""

import magenta.common.beam_search
import magenta.common.concurrency
import magenta.common.nade
import magenta.common.sequence_example_lib
import magenta.common.state_util
import magenta.common.testing_lib
import magenta.common.tf_utils
import magenta.music.abc_parser
import magenta.music.audio_io
import magenta.music.chord_symbols_lib
import magenta.music.chords_encoder_decoder
import magenta.music.chords_lib
import magenta.music.constants
import magenta.music.drums_encoder_decoder
import magenta.music.drums_lib
import magenta.music.encoder_decoder
import magenta.music.events_lib
import magenta.music.lead_sheets_lib
import magenta.music.melodies_lib
import magenta.music.melody_encoder_decoder
import magenta.music.midi_io
import magenta.music.midi_synth
import magenta.music.model
import magenta.music.musicxml_parser
import magenta.music.musicxml_reader
import magenta.music.note_sequence_io
import magenta.music.notebook_utils
import magenta.music.performance_encoder_decoder
import magenta.music.performance_lib
import magenta.music.pianoroll_encoder_decoder
import magenta.music.pianoroll_lib
import magenta.music.sequence_generator
import magenta.music.sequence_generator_bundle
import magenta.music.sequences_lib
import magenta.music.testing_lib
import magenta.pipelines.dag_pipeline
import magenta.pipelines.drum_pipelines
import magenta.pipelines.lead_sheet_pipelines
import magenta.pipelines.melody_pipelines
import magenta.pipelines.note_sequence_pipelines
import magenta.pipelines.pipeline
import magenta.pipelines.pipelines_common
import magenta.pipelines.statistics
import magenta.protobuf.generator_pb2
import magenta.protobuf.music_pb2
import magenta.version

from magenta.version import __version__
