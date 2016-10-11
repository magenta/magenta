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

"""Imports objects from music modules into the top-level music namespace."""

from constants import *  # pylint: disable=wildcard-import

from melodies_lib import BadNoteException
from melodies_lib import extract_melodies
from melodies_lib import Melody
from melodies_lib import MelodyEncoderDecoder
from melodies_lib import midi_file_to_melody
from melodies_lib import OneHotMelodyEncoderDecoder
from melodies_lib import PolyphonicMelodyException

from midi_io import midi_file_to_sequence_proto
from midi_io import midi_to_sequence_proto
from midi_io import MIDIConversionError
from midi_io import sequence_proto_to_midi_file
from midi_io import sequence_proto_to_pretty_midi

from midi_synth import fluidsynth
from midi_synth import synthesize

from notebook_utils import play_sequence

from sequence_generator import BaseSequenceGenerator
from sequence_generator import SequenceGeneratorException

from sequence_generator_bundle import GeneratorBundleParseException
from sequence_generator_bundle import read_bundle_file

from sequences_lib import BadTimeSignatureException
from sequences_lib import MultipleTimeSignatureException
from sequences_lib import NegativeTimeException
from sequences_lib import QuantizedSequence
