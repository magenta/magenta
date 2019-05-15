# JSB-Chorales-dataset
Dataset for JSB Chorales at different temporal resolutions, with train, validation, test split from [Boulanger-Lewandowski (2012)][nicolas-data].

Three temporal resolutions are provided: quarter, 8th, 16th. These "quantizations" are created by retaining the pitches heard on the specified temporal grid.  Boulanger-Lewandowski (2012) uses a temporal resolution of quarter notes.

This dataset currently does not encode fermatas and also does not distinguish between held and repeated notes.

## To load the data:
```python
import cPickle as pickle
with open('jsb-chorales-16th.pkl', 'wb') as p:
    data = pickle.load(p)
```

From Boulanger-Lewandowski (2012): "This will load a dictionary with 'train', 'valid' and 'test' keys, with the corresponding values being a list of sequences. Each sequence is itself a list of time steps, and each time step is a list of the non-zero elements in the piano-roll at this instant (in MIDI note numbers, between 21 and 108 inclusive)".

Additionally, the file `Jsb16thSeparated.npz` contains the same data in the format used in Coconet ([Huang & Cooijmans et al. ,2017][coconet]).  This format is like the above format except that at each time step there are exactly four numbers; one pitch for each of the SATB voices. If a voice is silent at a given time step, its pitch is NaN. This file was created based on the data included with the source code from https://tardis.ed.ac.uk/~moray/harmony/, but it contains the same data with the same train/valid/test split as the Boulanger-Lewandowski files.

## References:
Boulanger-Lewandowski, N., Vincent, P., & Bengio, Y. (2012). Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription. Proceedings of the 29th International Conference on Machine Learning (ICML-12), 1159–1166.

[nicolas-data]: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
[coconet]: https://ismir2017.smcnus.org/wp-content/uploads/2017/10/187_Paper.pdf
