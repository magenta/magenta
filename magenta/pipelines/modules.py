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

from magenta.lib import melodies_lib
from magenta.lib import sequences_lib
from magenta.protobuf import music_pb2

# Modules inherit base class Module.
class Module(object):
  
  # `input_type` can be an object, a tuple of objects,
  # or a dict of name to object pairs.
  input_type = None

  # `output_type` can be an object, a tuple of objects,
  # or a dict of name to object pairs.
  output_type = None

  def __init__(self, **settings_dict):
    """Module constructor. Pass Module's settings in here."""
    pass

  def transform(self, input):
    """Run this Module's transformation from input to output.

    Args:
      input: An instance of `input_type`. If `input_type` is a tuple
        of objects (object_0, object_1, ...) then input will be a tuple
        of instances (object_0(), object_1(), ...). If `input_type` is
        a dict of name to object pairs
        {"name_0": object_0, "name_1": object_1, ...} then input will be
        a dict of instances
        {"name_0": object_0(), "name_1": object_1(), ...}.

    Returns:
      A list of instances, tuples of instances, or dicts of name to
        instance pairs depending on `output_type`. See `input` docs.
    """
    pass

  # Returns a dict of stat name to counter or histogram pairs.
  def get_stats(self):
    """Produces a dict of stats after transform is called.

    Returns:
      A dictionary of stat name to state value pairs. Stat values can be
        counters or histograms.
    """
    return {}


class Quantizer(Module):
  input_type = music_pb2.NoteSequence
  output_type = sequences_lib.QuantizedSequence

  def __init__(self, steps_per_beat=4):
    super(Quantizer, self).__init__()
    self.steps_per_beat = steps_per_beat

  def transform(self, note_sequence):
    quantized_sequence = sequences_lib.QuantizedSequence()
    quantized_sequence.from_note_sequence(note_sequence, self.steps_per_beat)
    return [quantized_sequence]


class MelodyExtractor(Module):
  input_type = sequences_lib.QuantizedSequence
  output_type = melodies_lib.Melody

  def __init__(self, min_bars=7, min_unique_pitches=5, gap_bars=1.0):
    super(MelodyExtractor, self).__init__()
    self.min_bars = min_bars
    self.min_unique_pitches = min_unique_pitches
    self.gap_bars = gap_bars

  def transform(self, quantized_sequence):
    return melodies_lib.extract_melodies(quantized_sequence, min_bars=self.min_bars, min_unique_pitches=self.min_unique_pitches, gap_bars=self.gap_bars)