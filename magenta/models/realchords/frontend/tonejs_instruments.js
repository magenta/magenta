/**
 * @fileoverview A sample library and quick-loader for tone.js
 *
 * @author N.P. Brosowsky (nbrosowsky@gmail.com)
 * https://github.com/nbrosowsky/tonejs-instruments
 *
 * @author Yusong Wu
 * (yusongw@google.com) https://lukewys.github.io/PianorollVis.js
 * @suppress {visibility}
 */

let SampleLibrary = {
  minify: false,
  ext: '.[mp3|ogg]',  // use setExt to change the extensions on all files // do
                      // not change this variable //
  baseUrl: './samples/',
  list: [
    'bass-electric',   'bassoon',      'cello',       'clarinet',
    'contrabass',      'flute',        'french-horn', 'guitar-acoustic',
    'guitar-electric', 'guitar-nylon', 'harmonium',   'harp',
    'organ',           'piano',        'saxophone',   'trombone',
    'trumpet',         'tuba',         'violin',      'xylophone',
    'marimba',         'metronome',    'bell'
  ],
  onload: null,

  setExt: function(newExt) {
    for (let i = 0; i <= this.list.length - 1; i++) {
      for (let property in this[this.list[i]]) {
        this[this.list[i]][property] =
            this[this.list[i]][property].replace(this.ext, newExt);
      }
    }
    this.ext = newExt;
    return console.log('sample extensions set to ' + this.ext);
  },

  load: function(arg) {
    let t, rt, i;
    (arg) ? t = arg : t = {};
    t.instruments = t.instruments || this.list;
    t.baseUrl = t.baseUrl || this.baseUrl;
    t.onload = t.onload || this.onload;

    // update extensions if arg given
    if (t.ext) {
      if (t.ext != this.ext) {
        this.setExt(t.ext);
      }
      t.ext = this.ext;
    }

    rt = {};

    // if an array of instruments is passed...
    if (Array.isArray(t.instruments)) {
      for (i = 0; i <= t.instruments.length - 1; i++) {
        let newT = this[t.instruments[i]];
        // Minimize the number of samples to load
        if (this.minify === true || t.minify === true) {
          let minBy = 1;
          if (Object.keys(newT).length >= 17) {
            minBy = 2;
          }
          if (Object.keys(newT).length >= 33) {
            minBy = 4;
          }
          if (Object.keys(newT).length >= 49) {
            minBy = 6;
          }

          let filtered = Object.keys(newT).filter(function(_, i) {
            return i % minBy != 0;
          });
          filtered.forEach(function(f) {
            delete newT[f];
          });
        }

        rt[t.instruments[i]] = new Tone.Sampler(
            newT,
            {baseUrl: t.baseUrl + t.instruments[i] + '/', onload: t.onload});
      }

      return rt;

      // if a single instrument name is passed...
    } else {
      let newT = this[t.instruments];

      // Minimize the number of samples to load
      if (this.minify === true || t.minify === true) {
        let minBy = 1;
        if (Object.keys(newT).length >= 17) {
          minBy = 2;
        }
        if (Object.keys(newT).length >= 33) {
          minBy = 4;
        }
        if (Object.keys(newT).length >= 49) {
          minBy = 6;
        }

        let filtered = Object.keys(newT).filter(function(_, i) {
          return i % minBy != 0;
        });
        filtered.forEach(function(f) {
          delete newT[f];
        });
      }

      let s = new Tone.Sampler(
          newT, {baseUrl: t.baseUrl + t.instruments + '/', onload: t.onload});

      return s;
    }
  },

  'bass-electric': {
    'A#1': 'As1.[mp3|ogg]',
    'A#2': 'As2.[mp3|ogg]',
    'A#3': 'As3.[mp3|ogg]',
    'A#4': 'As4.[mp3|ogg]',
    'C#1': 'Cs1.[mp3|ogg]',
    'C#2': 'Cs2.[mp3|ogg]',
    'C#3': 'Cs3.[mp3|ogg]',
    'C#4': 'Cs4.[mp3|ogg]',
    'E1': 'E1.[mp3|ogg]',
    'E2': 'E2.[mp3|ogg]',
    'E3': 'E3.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'G1': 'G1.[mp3|ogg]',
    'G2': 'G2.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]'
  },

  'bassoon': {
    'A4': 'A4.[mp3|ogg]',
    'C3': 'C3.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'G2': 'G2.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]'

  },

  'cello': {
    'E3': 'E3.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'F2': 'F2.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]',
    'F4': 'F4.[mp3|ogg]',
    'F#3': 'Fs3.[mp3|ogg]',
    'F#4': 'Fs4.[mp3|ogg]',
    'G2': 'G2.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'G#2': 'Gs2.[mp3|ogg]',
    'G#3': 'Gs3.[mp3|ogg]',
    'G#4': 'Gs4.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A#2': 'As2.[mp3|ogg]',
    'A#3': 'As3.[mp3|ogg]',
    'B2': 'B2.[mp3|ogg]',
    'B3': 'B3.[mp3|ogg]',
    'B4': 'B4.[mp3|ogg]',
    'C2': 'C2.[mp3|ogg]',
    'C3': 'C3.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C#3': 'Cs3.[mp3|ogg]',
    'C#4': 'Cs4.[mp3|ogg]',
    'D2': 'D2.[mp3|ogg]',
    'D3': 'D3.[mp3|ogg]',
    'D4': 'D4.[mp3|ogg]',
    'D#2': 'Ds2.[mp3|ogg]',
    'D#3': 'Ds3.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]',
    'E2': 'E2.[mp3|ogg]'

  },

  'clarinet': {
    'D4': 'D4.[mp3|ogg]',
    'D5': 'D5.[mp3|ogg]',
    'D6': 'D6.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]',
    'F4': 'F4.[mp3|ogg]',
    'F5': 'F5.[mp3|ogg]',
    'F#6': 'Fs6.[mp3|ogg]',
    'A#3': 'As3.[mp3|ogg]',
    'A#4': 'As4.[mp3|ogg]',
    'A#5': 'As5.[mp3|ogg]',
    'D3': 'D3.[mp3|ogg]'

  },

  'contrabass': {
    'C2': 'C2.[mp3|ogg]',
    'C#3': 'Cs3.[mp3|ogg]',
    'D2': 'D2.[mp3|ogg]',
    'E2': 'E2.[mp3|ogg]',
    'E3': 'E3.[mp3|ogg]',
    'F#1': 'Fs1.[mp3|ogg]',
    'F#2': 'Fs2.[mp3|ogg]',
    'G1': 'G1.[mp3|ogg]',
    'G#2': 'Gs2.[mp3|ogg]',
    'G#3': 'Gs3.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A#1': 'As1.[mp3|ogg]',
    'B3': 'B3.[mp3|ogg]'

  },

  'flute': {
    'A6': 'A6.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C6': 'C6.[mp3|ogg]',
    'C7': 'C7.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'E5': 'E5.[mp3|ogg]',
    'E6': 'E6.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A5': 'A5.[mp3|ogg]'

  },

  'french-horn': {
    'D3': 'D3.[mp3|ogg]',
    'D5': 'D5.[mp3|ogg]',
    'D#2': 'Ds2.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]',
    'F5': 'F5.[mp3|ogg]',
    'G2': 'G2.[mp3|ogg]',
    'A1': 'A1.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'C2': 'C2.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',

  },

  'guitar-acoustic': {
    'F4': 'F4.[mp3|ogg]',
    'F#2': 'Fs2.[mp3|ogg]',
    'F#3': 'Fs3.[mp3|ogg]',
    'F#4': 'Fs4.[mp3|ogg]',
    'G2': 'G2.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'G#2': 'Gs2.[mp3|ogg]',
    'G#3': 'Gs3.[mp3|ogg]',
    'G#4': 'Gs4.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A#2': 'As2.[mp3|ogg]',
    'A#3': 'As3.[mp3|ogg]',
    'A#4': 'As4.[mp3|ogg]',
    'B2': 'B2.[mp3|ogg]',
    'B3': 'B3.[mp3|ogg]',
    'B4': 'B4.[mp3|ogg]',
    'C3': 'C3.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C#3': 'Cs3.[mp3|ogg]',
    'C#4': 'Cs4.[mp3|ogg]',
    'C#5': 'Cs5.[mp3|ogg]',
    'D2': 'D2.[mp3|ogg]',
    'D3': 'D3.[mp3|ogg]',
    'D4': 'D4.[mp3|ogg]',
    'D5': 'D5.[mp3|ogg]',
    'D#2': 'Ds2.[mp3|ogg]',
    'D#3': 'Ds3.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]',
    'E2': 'E2.[mp3|ogg]',
    'E3': 'E3.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'F2': 'F2.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]'

  },


  'guitar-electric': {
    'D#3': 'Ds3.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]',
    'D#5': 'Ds5.[mp3|ogg]',
    'E2': 'E2.[mp3|ogg]',
    'F#2': 'Fs2.[mp3|ogg]',
    'F#3': 'Fs3.[mp3|ogg]',
    'F#4': 'Fs4.[mp3|ogg]',
    'F#5': 'Fs5.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A5': 'A5.[mp3|ogg]',
    'C3': 'C3.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C6': 'C6.[mp3|ogg]',
    'C#2': 'Cs2.[mp3|ogg]'
  },

  'guitar-nylon': {
    'F#2': 'Fs2.[mp3|ogg]',
    'F#3': 'Fs3.[mp3|ogg]',
    'F#4': 'Fs4.[mp3|ogg]',
    'F#5': 'Fs5.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G5': 'G3.[mp3|ogg]',
    'G#2': 'Gs2.[mp3|ogg]',
    'G#4': 'Gs4.[mp3|ogg]',
    'G#5': 'Gs5.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A5': 'A5.[mp3|ogg]',
    'A#5': 'As5.[mp3|ogg]',
    'B1': 'B1.[mp3|ogg]',
    'B2': 'B2.[mp3|ogg]',
    'B3': 'B3.[mp3|ogg]',
    'B4': 'B4.[mp3|ogg]',
    'C#3': 'Cs3.[mp3|ogg]',
    'C#4': 'Cs4.[mp3|ogg]',
    'C#5': 'Cs5.[mp3|ogg]',
    'D2': 'D2.[mp3|ogg]',
    'D3': 'D3.[mp3|ogg]',
    'D5': 'D5.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]',
    'E2': 'E2.[mp3|ogg]',
    'E3': 'E3.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'E5': 'E5.[mp3|ogg]'
  },


  'harmonium': {
    'C2': 'C2.[mp3|ogg]',
    'C3': 'C3.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C#2': 'Cs2.[mp3|ogg]',
    'C#3': 'Cs3.[mp3|ogg]',
    'C#4': 'Cs4.[mp3|ogg]',
    'C#5': 'Cs5.[mp3|ogg]',
    'D2': 'D2.[mp3|ogg]',
    'D3': 'D3.[mp3|ogg]',
    'D4': 'D4.[mp3|ogg]',
    'D5': 'D5.[mp3|ogg]',
    'D#2': 'Ds2.[mp3|ogg]',
    'D#3': 'Ds3.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]',
    'E2': 'E2.[mp3|ogg]',
    'E3': 'E3.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'F2': 'F2.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]',
    'F4': 'F4.[mp3|ogg]',
    'F#2': 'Fs2.[mp3|ogg]',
    'F#3': 'Fs3.[mp3|ogg]',
    'G2': 'G2.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'G#2': 'Gs2.[mp3|ogg]',
    'G#3': 'Gs3.[mp3|ogg]',
    'G#4': 'Gs4.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A#2': 'As2.[mp3|ogg]',
    'A#3': 'As3.[mp3|ogg]',
    'A#4': 'As4.[mp3|ogg]'
  },

  'harp': {
    'C5': 'C5.[mp3|ogg]',
    'D2': 'D2.[mp3|ogg]',
    'D4': 'D4.[mp3|ogg]',
    'D6': 'D6.[mp3|ogg]',
    'D7': 'D7.[mp3|ogg]',
    'E1': 'E1.[mp3|ogg]',
    'E3': 'E3.[mp3|ogg]',
    'E5': 'E5.[mp3|ogg]',
    'F2': 'F2.[mp3|ogg]',
    'F4': 'F4.[mp3|ogg]',
    'F6': 'F6.[mp3|ogg]',
    'F7': 'F7.[mp3|ogg]',
    'G1': 'G1.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G5': 'G5.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A6': 'A6.[mp3|ogg]',
    'B1': 'B1.[mp3|ogg]',
    'B3': 'B3.[mp3|ogg]',
    'B5': 'B5.[mp3|ogg]',
    'B6': 'B6.[mp3|ogg]',
    'C3': 'C3.[mp3|ogg]'

  },

  'organ': {
    'C3': 'C3.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C6': 'C6.[mp3|ogg]',
    'D#1': 'Ds1.[mp3|ogg]',
    'D#2': 'Ds2.[mp3|ogg]',
    'D#3': 'Ds3.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]',
    'D#5': 'Ds5.[mp3|ogg]',
    'F#1': 'Fs1.[mp3|ogg]',
    'F#2': 'Fs2.[mp3|ogg]',
    'F#3': 'Fs3.[mp3|ogg]',
    'F#4': 'Fs4.[mp3|ogg]',
    'F#5': 'Fs5.[mp3|ogg]',
    'A1': 'A1.[mp3|ogg]',
    'A2': 'A2.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A5': 'A5.[mp3|ogg]',
    'C1': 'C1.[mp3|ogg]',
    'C2': 'C2.[mp3|ogg]'
  },

  'piano': {
    'A7': 'A7.wav',
    'A1': 'A1.wav',
    'A2': 'A2.wav',
    'A3': 'A3.wav',
    'A4': 'A4.wav',
    'A5': 'A5.wav',
    'A6': 'A6.wav',
    'A#7': 'As7.wav',
    'A#1': 'As1.wav',
    'A#2': 'As2.wav',
    'A#3': 'As3.wav',
    'A#4': 'As4.wav',
    'A#5': 'As5.wav',
    'A#6': 'As6.wav',
    'B7': 'B7.wav',
    'B1': 'B1.wav',
    'B2': 'B2.wav',
    'B3': 'B3.wav',
    'B4': 'B4.wav',
    'B5': 'B5.wav',
    'B6': 'B6.wav',
    'C1': 'C1.wav',
    'C2': 'C2.wav',
    'C3': 'C3.wav',
    'C4': 'C4.wav',
    'C5': 'C5.wav',
    'C6': 'C6.wav',
    'C7': 'C7.wav',
    'C#7': 'Cs7.wav',
    'C#1': 'Cs1.wav',
    'C#2': 'Cs2.wav',
    'C#3': 'Cs3.wav',
    'C#4': 'Cs4.wav',
    'C#5': 'Cs5.wav',
    'C#6': 'Cs6.wav',
    'D7': 'D7.wav',
    'D1': 'D1.wav',
    'D2': 'D2.wav',
    'D3': 'D3.wav',
    'D4': 'D4.wav',
    'D5': 'D5.wav',
    'D6': 'D6.wav',
    'D#7': 'Ds7.wav',
    'D#1': 'Ds1.wav',
    'D#2': 'Ds2.wav',
    'D#3': 'Ds3.wav',
    'D#4': 'Ds4.wav',
    'D#5': 'Ds5.wav',
    'D#6': 'Ds6.wav',
    'E7': 'E7.wav',
    'E1': 'E1.wav',
    'E2': 'E2.wav',
    'E3': 'E3.wav',
    'E4': 'E4.wav',
    'E5': 'E5.wav',
    'E6': 'E6.wav',
    'F7': 'F7.wav',
    'F1': 'F1.wav',
    'F2': 'F2.wav',
    'F3': 'F3.wav',
    'F4': 'F4.wav',
    'F5': 'F5.wav',
    'F6': 'F6.wav',
    'F#7': 'Fs7.wav',
    'F#1': 'Fs1.wav',
    'F#2': 'Fs2.wav',
    'F#3': 'Fs3.wav',
    'F#4': 'Fs4.wav',
    'F#5': 'Fs5.wav',
    'F#6': 'Fs6.wav',
    'G7': 'G7.wav',
    'G1': 'G1.wav',
    'G2': 'G2.wav',
    'G3': 'G3.wav',
    'G4': 'G4.wav',
    'G5': 'G5.wav',
    'G6': 'G6.wav',
    'G#7': 'Gs7.wav',
    'G#1': 'Gs1.wav',
    'G#2': 'Gs2.wav',
    'G#3': 'Gs3.wav',
    'G#4': 'Gs4.wav',
    'G#5': 'Gs5.wav',
    'G#6': 'Gs6.wav'
  },

  'saxophone': {
    'D#5': 'Ds5.[mp3|ogg]',
    'E3': 'E3.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'E5': 'E5.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]',
    'F4': 'F4.[mp3|ogg]',
    'F5': 'F5.[mp3|ogg]',
    'F#3': 'Fs3.[mp3|ogg]',
    'F#4': 'Fs4.[mp3|ogg]',
    'F#5': 'Fs5.[mp3|ogg]',
    'G3': 'G3.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'G5': 'G5.[mp3|ogg]',
    'G#3': 'Gs3.[mp3|ogg]',
    'G#4': 'Gs4.[mp3|ogg]',
    'G#5': 'Gs5.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A5': 'A5.[mp3|ogg]',
    'A#3': 'As3.[mp3|ogg]',
    'A#4': 'As4.[mp3|ogg]',
    'B3': 'B3.[mp3|ogg]',
    'B4': 'B4.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C#3': 'Cs3.[mp3|ogg]',
    'C#4': 'Cs4.[mp3|ogg]',
    'C#5': 'Cs5.[mp3|ogg]',
    'D3': 'D3.[mp3|ogg]',
    'D4': 'D4.[mp3|ogg]',
    'D5': 'D5.[mp3|ogg]',
    'D#3': 'Ds3.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]'

  },

  // 'trombone': {
  //     'A#3': 'As3.[mp3|ogg]',
  //     'C3': 'C3.[mp3|ogg]',
  //     'C4': 'C4.[mp3|ogg]',
  //     'C#2': 'Cs2.[mp3|ogg]',
  //     'C#4': 'Cs4.[mp3|ogg]',
  //     'D3': 'D3.[mp3|ogg]',
  //     'D4': 'D4.[mp3|ogg]',
  //     'D#2': 'Ds2.[mp3|ogg]',
  //     'D#3': 'Ds3.[mp3|ogg]',
  //     'D#4': 'Ds4.[mp3|ogg]',
  //     'F2': 'F2.[mp3|ogg]',
  //     'F3': 'F3.[mp3|ogg]',
  //     'F4': 'F4.[mp3|ogg]',
  //     'G#2': 'Gs2.[mp3|ogg]',
  //     'G#3': 'Gs3.[mp3|ogg]',
  //     'A#1': 'As1.[mp3|ogg]',
  //     'A#2': 'As2.[mp3|ogg]'
  //
  // },

  'trumpet': {
    'C6': 'C6.[mp3|ogg]',
    'D5': 'D5.[mp3|ogg]',
    'D#4': 'Ds4.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]',
    'F4': 'F4.[mp3|ogg]',
    'F5': 'F5.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'A3': 'A3.[mp3|ogg]',
    'A5': 'A5.[mp3|ogg]',
    'A#4': 'As4.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]'

  },

  'tuba': {
    'A#2': 'As2.[mp3|ogg]',
    'A#3': 'As3.[mp3|ogg]',
    'D3': 'D3.[mp3|ogg]',
    'D4': 'D4.[mp3|ogg]',
    'D#2': 'Ds2.[mp3|ogg]',
    'F1': 'F1.[mp3|ogg]',
    'F2': 'F2.[mp3|ogg]',
    'F3': 'F3.[mp3|ogg]',
    'A#1': 'As1.[mp3|ogg]'

  },

  'violin': {
    'A3': 'A3.[mp3|ogg]',
    'A4': 'A4.[mp3|ogg]',
    'A5': 'A5.[mp3|ogg]',
    'A6': 'A6.[mp3|ogg]',
    'C4': 'C4.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C6': 'C6.[mp3|ogg]',
    'C7': 'C7.[mp3|ogg]',
    'E4': 'E4.[mp3|ogg]',
    'E5': 'E5.[mp3|ogg]',
    'E6': 'E6.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'G5': 'G5.[mp3|ogg]',
    'G6': 'G6.[mp3|ogg]'

  },

  'xylophone': {
    'C8': 'C8.[mp3|ogg]',
    'G4': 'G4.[mp3|ogg]',
    'G5': 'G5.[mp3|ogg]',
    'G6': 'G6.[mp3|ogg]',
    'G7': 'G7.[mp3|ogg]',
    'C5': 'C5.[mp3|ogg]',
    'C6': 'C6.[mp3|ogg]',
    'C7': 'C7.[mp3|ogg]'

  },
  'marimba': {
    'D2': 'D2.wav',
    'D3': 'D3.wav',
    'D4': 'D4.wav',
    'D5': 'D5.wav',
    'D6': 'D6.wav',
    'D7': 'D7.wav',
    'E2': 'E2.wav',
    'E3': 'E3.wav',
    'E4': 'E4.wav',
    'E5': 'E5.wav',
    'E6': 'E6.wav',
    'E7': 'E7.wav',
    'F2': 'F2.wav',
    'F3': 'F3.wav',
    'F4': 'F4.wav',
    'F5': 'F5.wav',
    'F6': 'F6.wav',
    'F7': 'F7.wav',
    'F#2': 'Fs2.wav',
    'F#3': 'Fs3.wav',
    'F#4': 'Fs4.wav',
    'F#5': 'Fs5.wav',
    'F#6': 'Fs6.wav',
    'F#7': 'Fs7.wav',
    'G2': 'G2.wav',
    'G3': 'G3.wav',
    'G4': 'G4.wav',
    'G5': 'G5.wav',
    'G6': 'G6.wav',
    'G7': 'G7.wav',
    'G#2': 'Gs2.wav',
    'G#3': 'Gs3.wav',
    'G#4': 'Gs4.wav',
    'G#5': 'Gs5.wav',
    'G#6': 'Gs6.wav',
    'G#7': 'Gs7.wav',
    'A2': 'A2.wav',
    'A3': 'A3.wav',
    'A4': 'A4.wav',
    'A5': 'A5.wav',
    'A6': 'A6.wav',
    'A7': 'A7.wav',
    'A#2': 'As2.wav',
    'A#3': 'As3.wav',
    'A#4': 'As4.wav',
    'A#5': 'As5.wav',
    'A#6': 'As6.wav',
    'A#7': 'As7.wav',
    'B2': 'B2.wav',
    'B3': 'B3.wav',
    'B4': 'B4.wav',
    'B5': 'B5.wav',
    'B6': 'B6.wav',
    'B7': 'B7.wav',
    'C2': 'C2.wav',
    'C3': 'C3.wav',
    'C4': 'C4.wav',
    'C5': 'C5.wav',
    'C6': 'C6.wav',
    'C7': 'C7.wav',
    'C8': 'C8.wav',
    'C#2': 'Cs2.wav',
    'C#3': 'Cs3.wav',
    'C#4': 'Cs4.wav',
    'C#5': 'Cs5.wav',
    'C#6': 'Cs6.wav',
    'C#7': 'Cs7.wav',
    'D#2': 'Ds2.wav',
    'D#3': 'Ds3.wav',
    'D#4': 'Ds4.wav',
    'D#5': 'Ds5.wav',
    'D#6': 'Ds6.wav',
    'D#7': 'Ds7.wav'
  },

  // from
  // https://github.com/mrmrmrfinch/BachDuet-WebGUI/tree/main/public/audio/samples/metronome
  // original "C0.mp3" --> "C1.mp3", original "Cs0.mp3" --> "C0.mp3"
  // Changed file name because the original "C0.mp3" is higher than the original
  // "Cs0.mp3"
  'metronome': {
    'C1': 'C1.wav',
    'C0': 'C0.wav',
    'C2': 'C2.wav',
  },

  // https://freesound.org/people/Seidhepriest/sounds/191959/
  'bell': {
    'C0': 'bell.wav',
  }
};
