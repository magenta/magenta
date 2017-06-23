import os
import urllib

# This script downloads these .mag files if not already present.
mag_files = [
    'http://download.magenta.tensorflow.org/models/attention_rnn.mag',
    'http://download.magenta.tensorflow.org/models/pianoroll_rnn_nade.mag',
    'http://download.magenta.tensorflow.org/models/drum_kit_rnn.mag',
]

for mag_file in mag_files:
  output_file = mag_file.split('/')[-1]
  if os.path.exists(output_file):
    print 'File %s already present' % mag_file
  else:
    print 'Writing %s to %s' % (mag_file, output_file)
    urlopener = urllib.URLopener()
    urlopener.retrieve(mag_file, output_file)
