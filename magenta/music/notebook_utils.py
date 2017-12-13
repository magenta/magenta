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
"""Python functions which run only within a Jupyter or Colab notebook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import collections
from io import BytesIO
import os

# internal imports

import bokeh
import bokeh.plotting
from IPython import display
import numpy as np
import pandas as pd
from scipy.io import wavfile
from six.moves import urllib

from magenta.music import midi_synth

_DEFAULT_SAMPLE_RATE = 44100
_play_id = 0  # Used for ephemeral colab_play.


def colab_play(array_of_floats, sample_rate, ephemeral=True, autoplay=False):
  """Creates an HTML5 audio widget to play a sound in Colab.

  This function should only be called from a Colab notebook.

  Args:
    array_of_floats: A 1D or 2D array-like container of float sound
      samples. Values outside of the range [-1, 1] will be clipped.
    sample_rate: Sample rate in samples per second.
    ephemeral: If set to True, the widget will be ephemeral, and disappear
      on reload (and it won't be counted against realtime document size).
    autoplay: If True, automatically start playing the sound when the
      widget is rendered.
  """
  from google.colab.output import _js_builder as js  # pylint: disable=g-import-not-at-top,protected-access

  normalizer = float(np.iinfo(np.int16).max)
  array_of_ints = np.array(
      np.asarray(array_of_floats) * normalizer, dtype=np.int16)
  memfile = BytesIO()
  wavfile.write(memfile, sample_rate, array_of_ints)
  html = """<audio controls {autoplay}>
              <source controls src="data:audio/wav;base64,{base64_wavfile}"
              type="audio/wav" />
              Your browser does not support the audio element.
            </audio>"""
  html = html.format(
      autoplay='autoplay' if autoplay else '',
      base64_wavfile=base64.encodestring(memfile.getvalue()))
  memfile.close()
  global _play_id
  _play_id += 1
  if ephemeral:
    element = 'id_%s' % _play_id
    display.display(display.HTML('<div id="%s"> </div>' % element))
    js.Js('document', mode=js.EVAL).getElementById(element).innerHTML = html
  else:
    display.display(display.HTML(html))


def play_sequence(sequence,
                  synth=midi_synth.synthesize,
                  sample_rate=_DEFAULT_SAMPLE_RATE,
                  colab_ephemeral=True,
                  **synth_args):
  """Creates an interactive player for a synthesized note sequence.

  This function should only be called from a Jupyter or Colab notebook.

  Args:
    sequence: A music_pb2.NoteSequence to synthesize and play.
    synth: A synthesis function that takes a sequence and sample rate as input.
    sample_rate: The sample rate at which to synthesize.
    colab_ephemeral: If set to True, the widget will be ephemeral in Colab, and
      disappear on reload (and it won't be counted against realtime document
      size).
    **synth_args: Additional keyword arguments to pass to the synth function.
  """
  array_of_floats = synth(sequence, sample_rate=sample_rate, **synth_args)

  try:
    import google.colab  # pylint: disable=unused-import,unused-variable,g-import-not-at-top
    colab_play(array_of_floats, sample_rate, colab_ephemeral)
  except ImportError:
    display.display(display.Audio(array_of_floats, rate=sample_rate))


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
      pd_dict['fill_alpha'].append(note.velocity / 128.0)
      pd_dict['instrument'].append(note.instrument)
      pd_dict['program'].append(note.program)

    # If no velocity differences are found, set alpha to 1.0.
    if np.max(pd_dict['velocity']) == np.min(pd_dict['velocity']):
      pd_dict['fill_alpha'] = [1.0] * len(pd_dict['fill_alpha'])

    return pd.DataFrame(pd_dict)

  # These are hard-coded reasonable values, but the user can override them
  # by updating the figure if need be.
  fig = bokeh.plotting.figure(
      tools='hover,pan,box_zoom,reset,previewsave')
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
             line_color='black', fill_color=color,
             fill_alpha='fill_alpha', source=source)
  fig.select(dict(type=bokeh.models.HoverTool)).tooltips = (
      {'pitch': '@pitch',
       'program': '@program',
       'velo': '@velocity',
       'duration': '@duration',
       'start_time': '@start_time',
       'end_time': '@end_time',
       'velocity': '@velocity',
       'fill_alpha': '@fill_alpha'})

  if show_figure:
    bokeh.plotting.output_notebook()
    bokeh.plotting.show(fig)
    return None
  return fig


def download_bundle(bundle_name, target_dir, force_reload=False):
  """Downloads a Magenta bundle to target directory.

  Args:
     bundle_name: A string Magenta bundle name to download.
     target_dir: A string local directory in which to write the bundle.
     force_reload: A boolean that when True, reloads the bundle even if present.
  """
  bundle_target = os.path.join(target_dir, bundle_name)
  if not os.path.exists(bundle_target) or force_reload:
    response = urllib.request.urlopen(
        'http://download.magenta.tensorflow.org/models/%s' % bundle_name)
    data = response.read()
    local_file = open(bundle_target, 'wb')
    local_file.write(data)
    local_file.close()
