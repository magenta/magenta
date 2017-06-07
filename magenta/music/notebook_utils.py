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
"""Python functions which run only within a Jupyter notebook."""

import collections

# internal imports

import bokeh
import bokeh.plotting
import IPython
import pandas as pd

from magenta.music import midi_synth

_DEFAULT_SAMPLE_RATE = 44100


def play_sequence(sequence,
                  synth=midi_synth.synthesize,
                  sample_rate=_DEFAULT_SAMPLE_RATE,
                  **synth_args):
  """Creates an interactive player for a synthesized note sequence.

  This function should only be called from a Jupyter notebook.

  Args:
    sequence: A music_pb2.NoteSequence to synthesize and play.
    synth: A synthesis function that takes a sequence and sample rate as input.
    sample_rate: The sample rate at which to synthesize.
    **synth_args: Additional keyword arguments to pass to the synth function.
  """

  array_of_floats = synth(sequence, sample_rate=sample_rate, **synth_args)
  IPython.display.display(IPython.display.Audio(array_of_floats,
                                                rate=sample_rate))


def plot_sequence(sequence,
                  show_figure=True):
  """Creates an interactive pianoroll for a tensorflow.magenta.NoteSequence.

  Example usage: plot a random melody.
    sequence = mm.Melody(np.random.randint(36, 72, 30)).to_sequence()
    bokeh_pianoroll(sequence)

  Args:
     sequence: A tensorflow.magenta.NoteSequence.
     show_figure: A boolean indicating whether or not to show the figure.

  Returns:
     If show_figure is False, a Bokeh figure; otherwise None.
  """

  def _sequence_to_pandas_dataframe(sequence):
    """Generates a pandas dataframe from a sequence."""
    pd_dict = collections.defaultdict(list)
    for note in sequence.notes:
      pd_dict['start_time'].append(note.start_time)
      pd_dict['end_time'].append(note.end_time)
      pd_dict['duration'].append(note.end_time - note.start_time)
      pd_dict['pitch'].append(note.pitch)
      pd_dict['bottom'].append(note.pitch - 0.4)
      pd_dict['top'].append(note.pitch + 0.4)
      pd_dict['velocity'].append(note.velocity)
      pd_dict['instrument'].append(note.instrument)
      pd_dict['program'].append(note.program)
    return pd.DataFrame(pd_dict)

  # These are hard-coded reasonable values, but the user can override them
  # by updating the figure if need be.
  fig = bokeh.plotting.figure(
      tools='hover,pan,resize,box_zoom,reset,previewsave')
  fig.plot_width = 500
  fig.plot_height = 200
  fig.xaxis.axis_label = 'time (sec)'
  fig.yaxis.axis_label = 'pitch (MIDI)'
  fig.yaxis.ticker = bokeh.models.SingleIntervalTicker(interval=12)
  fig.ygrid.ticker = bokeh.models.SingleIntervalTicker(interval=12)
  # Pick indexes that are maximally different in Spectral8 colormap.
  spectral_color_indexes = [7, 0, 6, 1, 5, 2, 3]

  # Create a Pandas dataframe and group it by instrument.
  dataframe = _sequence_to_pandas_dataframe(sequence)
  instruments = sorted(set(dataframe['instrument']))
  grouped_dataframe = dataframe.groupby('instrument')
  for counter, instrument in enumerate(instruments):
    instrument_df = grouped_dataframe.get_group(instrument)
    color_idx = spectral_color_indexes[counter % len(spectral_color_indexes)]
    color = bokeh.palettes.Spectral8[color_idx]
    source = bokeh.plotting.ColumnDataSource(instrument_df)
    fig.quad(top='top', bottom='bottom', left='start_time', right='end_time',
             line_color='black', fill_color=color, source=source)
  fig.select(dict(type=bokeh.models.HoverTool)).tooltips = (
      {'pitch': '@pitch',
       'program': '@program',
       'velo': '@velocity',
       'duration': '@duration',
       'start_time': '@start_time',
       'end_time': '@end_time'})

  if show_figure:
    bokeh.plotting.output_notebook()
    bokeh.plotting.show(fig)
    return None
  return fig
