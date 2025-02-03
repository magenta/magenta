/**
 * @fileoverview Agent interaction, live session control, piano
 * interaction, audio, and UI.
 *
 * @author Alex Scarlatos (scarlatos@google.com)
 *
 * @author Yusong Wu (yusongw@google.com)
 * https://lukewys.github.io/PianorollVis.js
 * @suppress {visibility}
 */

let visual, chordSynth, melodySynth, metronome, webMIDIInputs, instrumentMap;
let mouseDown = false, compKeyboardOctave = 4, keysToNotes, heldNotes = {};
let metronomeStatus = false, metronomeEvents = [], metronomeFreq = 'beat';
let curSession, lastSession, recorder, curAudioRecording;
let playBtn, metronomeBtn, bpmInput, timeSigInput, metronomeFreqBtn,
    interfaceSelect, liveSessionBtn, temperatureInput, silenceInput,
    lookaheadInput, commitaheadInput, modelSelect, chordInstSelect,
    melodyInstSelect, showChordsCheck, downloadSessionCheck, metronomeCheck;

const fpb = 4;  // Frames per beat
const chordVelocity = 0.5;
const melodyVelocity = 0.5;
const metronomeVelocity = 0.5;
/** @enum {number} */
const compKeyOctaveRange = [2, 6];

const DEFAULTS = {
  bpm: window.location.hostname === 'localhost' ? 30 : 80,
  timeSignature: 4,
  temperature: 0.5,
  silence: window.location.hostname === 'localhost' ? 4 : 8,
  lookahead: window.location.hostname === 'localhost' ? 2 : 4,
  commitahead: window.location.hostname === 'localhost' ? 2 : 4,
  chordInstrument: 'Piano (Versilian)',
  melodyInstrument: 'Piano (Versilian)',
};

/** @enum {string} */
const pianoNotes = [
  'A0',  'A#0', 'B0',  'C1',  'C#1', 'D1',  'D#1', 'E1',  'F1',  'F#1', 'G1',
  'G#1', 'A1',  'A#1', 'B1',  'C2',  'C#2', 'D2',  'D#2', 'E2',  'F2',  'F#2',
  'G2',  'G#2', 'A2',  'A#2', 'B2',  'C3',  'C#3', 'D3',  'D#3', 'E3',  'F3',
  'F#3', 'G3',  'G#3', 'A3',  'A#3', 'B3',  'C4',  'C#4', 'D4',  'D#4', 'E4',
  'F4',  'F#4', 'G4',  'G#4', 'A4',  'A#4', 'B4',  'C5',  'C#5', 'D5',  'D#5',
  'E5',  'F5',  'F#5', 'G5',  'G#5', 'A5',  'A#5', 'B5',  'C6',  'C#6', 'D6',
  'D#6', 'E6',  'F6',  'F#6', 'G6',  'G#6', 'A6',  'A#6', 'B6',  'C7',  'C#7',
  'D7',  'D#7', 'E7',  'F7',  'F#7', 'G7',  'G#7', 'A7',  'A#7', 'B7',  'C8'
];

/** @enum {number} */
const noteToPitch =
    pianoNotes.reduce((prev, note, idx) => ({...prev, [note]: idx + 21}), {});

/** @enum {string} */
const pitchToNote =
    pianoNotes.reduce((prev, note, idx) => ({...prev, [idx + 21]: note}), {});

/**
 * Return if two arrays have the same elements
 * @param {Array!} arr1
 * @param {Array!} arr2
 * @return {boolean}
 */
function arraysEqual(arr1, arr2) {
  return arr1.length === arr2.length && arr1.every((el, i) => el === arr2[i]);
}

/**
 * Add input cleaning and an optional callback to a numeric input element
 * @param {Object!} inputEl
 * @param {callback!} callback
 */
function addNumericInputEventListener(inputEl, callback) {
  inputEl.addEventListener('input', event => {
    let floatVal = parseFloat(event.target.value);
    if (isNaN(floatVal)) {
      floatVal = inputEl.min || 0;
    }
    if (inputEl.min) {
      floatVal = Math.max(floatVal, inputEl.min);
    }
    if (inputEl.max) {
      floatVal = Math.min(floatVal, inputEl.max);
    }
    inputEl.value = floatVal;
    if (callback) {
      callback(floatVal);
    }
  });
}

/** Initialize components after loading is done */
function showMainScreen() {
  document.querySelector('.splash').hidden = true;
  document.querySelector('.loaded').hidden = false;

  Tone.context.lookAhead = 0;
  Tone.start();
  enableClickingInputs();
  enableKeyboardInputs();
  document.addEventListener('mousedown', () => {
    mouseDown = true;
  });
  document.addEventListener('mouseup', () => {
    mouseDown = false;
  });

  // Enable WEBMIDI.js and then prepare the input interfaces
  WebMidi.enable().then(enableMIDIInputs).catch(err => alert(err));
}

/** Download audio recording and JSON of the most recent session */
function saveSessionRecording() {
  if (!downloadSessionCheck.checked) {
    return;
  }
  for (const [data, type] of [
           [curAudioRecording, 'audio/ogg; codecs=opus'],
           [[JSON.stringify(lastSession)], 'text/plain']]) {
    const blob = new Blob(data, {type});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style = 'display: none';
    a.href = url;
    a.download =
        `GenJam Session - ${lastSession.startTime} ${lastSession.description}`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

/** Start or stop a live session */
function toggleLiveSession() {
  if (curSession) {
    lastSession = curSession;
    lastSession.description = `sic${showChordsCheck.checked}_bpm${
        bpmInput.value}_met${metronomeStatus}_la${lookaheadInput.value}_com${
        commitaheadInput.value}_sil${silenceInput.value}_temp${
        temperatureInput.value}_${modelSelect.value}`;
    curSession = undefined;
    liveSessionBtn.textContent = 'Start Live Session';
    showChordsCheck.disabled = false;
    bpmInput.disabled = false;
    timeSigInput.disabled = false;
    silenceInput.disabled = false;
    Tone.Transport.stop();
    Tone.Transport.cancel();
    stopMetronome();
    pianoNotes.forEach(note => chordSynth.triggerRelease(note));
    visual.clearScheduledNotes(0);
    visual.stopAllNotes();
    recorder.stop();
  } else {
    Tone.Transport.stop();
    Tone.Transport.start();
    if (metronomeCheck.checked) {
      startMetronome();
    }
    curAudioRecording = [];
    recorder.start();
    curSession = {
      startTime: Date.now(),
      startFrame: undefined,  // Frame relative to Transport start that first
                              // note is played
      noteHistory: [],        // All note hits and releases
      chordHistory: [],       // All chord pitch/symbol onsets with frames
      chordTokens: [],        // All chord tokens in frame format
      introSet: false,        // If model has generated intro section
    };
    liveSessionBtn.textContent = 'Stop Live Session';
    showChordsCheck.disabled = true;
    bpmInput.disabled = true;
    timeSigInput.disabled = true;
    silenceInput.disabled = true;
  }
}

/**
 * Set the lookahead value in beats
 * @param {number} lookahead
 */
function setLookahead(lookahead) {
  visual.setVisibleFrames(lookahead * fpb);
  commitaheadInput.max = lookahead;
  if (commitaheadInput.valueAsNumber > lookahead) {
    commitaheadInput.value = lookahead;
  }
}

/**
 * Get the number of lookahead frames
 * @return {number}
 */
function getLookaheadFrames() {
  return lookaheadInput.valueAsNumber * fpb;
}

/**
 * Get the number of commitahead frames
 * @return {number}
 */
function getCommitaheadFrames() {
  return commitaheadInput.valueAsNumber * fpb;
}

/**
 * Get the number of initial frames of silence
 * @return {number}
 */
function getSilenceFrames() {
  return silenceInput.valueAsNumber * fpb;
}

/**
 * Get the current frame of the session
 * Also set the start frame on the first call
 * @return {number}
 */
function getSessionCurrentFrame() {
  // Get current frame
  const softFrame = getTransportFrame();
  const frame = Math.round(softFrame);

  // Set start frame if needed, subtract modulo to always start on the beat
  if (!curSession.startFrame) {
    curSession.startFrame = frame - (frame % fpb);
  }

  return frame - curSession.startFrame;
}

/** Send session history as context to model to get new chord predictions */
async function syncWithServer() {
  // Exit loop if current session ended (in case ended during timeout)
  if (!curSession) {
    return;
  }

  // Send current frame and context to server for generation
  const curFrame = getSessionCurrentFrame();
  const result = await fetch(`${window.location.origin}/play`, {
    'method': 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: modelSelect.value,
      notes: curSession.noteHistory,
      chordTokens: curSession.chordTokens,
      frame: curFrame + 1,  // Request chords to play at the next frame
      lookahead: getLookaheadFrames(),
      commitahead: getCommitaheadFrames(),
      silenceTill: getSilenceFrames(),
      temperature: temperatureInput.valueAsNumber,
      introSet: curSession.introSet,
    })
  });
  const json = await result.json();

  // Exit loop if current session ended (in case ended during fetch)
  if (!curSession) {
    return;
  }

  // Schedule chords based on agent response
  processAgentAction(json);

  // Send next request to server, wait until next frame if haven't advanced yet
  if (curFrame === getSessionCurrentFrame()) {
    const fps = fpb * Tone.Transport.bpm.value / 60;
    setTimeout(syncWithServer, 1000 / fps);
  } else {
    syncWithServer();
  }
}

/**
 * Get chord pitches being played or held at the given frame
 * @param {number} targetFrame
 * @return {Array<number>!}
 */
function getChordPitchesAtFrame(targetFrame) {
  for (let i = curSession.chordHistory.length - 1; i >= 0; i--) {
    const {pitches, scheduleFrame} = curSession.chordHistory[i];
    if (scheduleFrame <= targetFrame) {
      return pitches;
    }
  }
  return [];
}

/**
 * Schedule to play chord pitches in the future
 * @param {Array<number>!} pitches
 * @param {number} frame - Frame (relative to ToneJS start) to schedule at
 * @param {number} time - ToneJS time to schedule at
 * @param {boolean} on - If the pitches should hit or release
 * @param {number=} alpha - Opacity of incoming chords on grid
 * @param {string=} symbol - Chord symbol to draw next to incoming chord on grid
 * @return {Array<number>!} ToneJS IDs of scheduled events
 */
function scheduleChordPitches(
    pitches, frame, time, on, alpha = 1.0, symbol = '') {
  let eventIDs = [];
  const bassPitch =
      pitches.reduce((lowest, pitch) => Math.min(lowest, pitch), 1000);
  pitches.forEach(pitch => {
    eventIDs.push(scheduleNote(pitchToNote[pitch], on, time));
    if (showChordsCheck.checked) {
      if (on) {
        visual.scheduleNoteOn(
            pitch, frame, 'lightBlue', alpha,
            pitch === bassPitch ? symbol : '');
      } else {
        visual.scheduleNoteOff(pitch, frame);
      }
    }
  });
  return eventIDs;
}

/**
 * Process response from chord model
 * @param {Object!} json
 * newChords - New chord predictions starting at frame
 * newChordTokens - Per-frame tokens for new chords
 * introChordTokens - Per-frame tokens for chords at beginning of session
 * frame - Frame new chords start at
 */
function processAgentAction(
    {newChords, newChordTokens, introChordTokens, frame}) {
  const curFrame = getTransportFrame();
  const targetFrame = frame + curSession.startFrame;
  const silenceFrame = getSilenceFrames() + curSession.startFrame;

  // Fill in new chord tokens and fill possible gaps
  for (let i = Math.min(frame, curSession.chordTokens.length);
       i < frame + newChordTokens.length; i++) {
    if (i < frame) {
      curSession.chordTokens[i] = -1;
    } else {
      curSession.chordTokens[i] = newChordTokens[i - frame];
    }
  }

  // Fill intro chord tokens when model passes them back
  //  (after post-hoc filling of initial silence frames)
  if (introChordTokens) {
    curSession.introSet = true;
    for (let i = 0; i < introChordTokens.length; i++) {
      curSession.chordTokens[i] = introChordTokens[i];
    }
  }

  // Cancel all scheduled events since will be replaced with fresher actions
  // Make sure to not clear before current frame or target frame
  const clearFrame = Math.max(targetFrame, curFrame);
  visual.clearScheduledNotes(clearFrame);
  for (let i = curSession.chordHistory.length - 1; i >= 0; i--) {
    if (curSession.chordHistory[i].scheduleFrame < clearFrame) {
      curSession.chordHistory = curSession.chordHistory.slice(0, i + 1);
      break;
    }
    curSession.chordHistory[i].eventIDs.forEach(
        eventID => Tone.Transport.clear(eventID));
  }

  // Schedule note hits and releases for sent frames
  newChords.forEach(([symbol, pitches, on], frameOffset) => {
    const scheduleFrame = targetFrame + frameOffset;
    const time = frameToTransportTime(scheduleFrame);
    console.log(
        'Play', symbol, on, 'at frame', scheduleFrame, ', cur frame', curFrame);

    // Don't schedule/record chords for frames that already happened
    // Otherwise could cause issues with keeping track of currently held pitches
    // Also don't schedule until after silence frames
    if (scheduleFrame < curFrame || scheduleFrame < silenceFrame) {
      return;
    }

    const isCommitted = frameOffset < getCommitaheadFrames();
    const prevFramePitches = getChordPitchesAtFrame(scheduleFrame - 1);

    // Onsets - release previous chord and play new chord (or rest)
    // Holds - release previous and play chord if different than previous
    //  (indicates model changed its mind about what chord to play)
    if (on || !arraysEqual(pitches, prevFramePitches)) {
      const offEventIDs =
          scheduleChordPitches(prevFramePitches, scheduleFrame, time, false);
      const onEventIDs = scheduleChordPitches(
          pitches, scheduleFrame, time, true, isCommitted ? 1.0 : 0.5, symbol);
      curSession.chordHistory.push({
        scheduleFrame,
        pitches,
        symbol,
        eventIDs: offEventIDs.concat(onEventIDs)
      });
    }
  });
}

/**
 * Get necessary info from server (model names) and then enable live sessions
 */
function establishServerConnection() {
  fetch(`${window.location.origin}/models`)
      .then(response => response.json())
      .then(json => {
        json.forEach(model => {
          const name = document.createTextNode(model);
          const option = document.createElement('option');
          option.append(name);
          option.value = model;
          modelSelect.appendChild(option);
        });
        console.log('Interactive agent is ready!');
        liveSessionBtn.disabled = false;
      });
}

/**
 * Play melody note, add to history and start live session loop if needed
 * @param {string} note
 * @param {number} velocity
 */
function playNote(note, velocity) {
  const keyIndex = noteToPitch[note];
  if (curSession) {
    const startLoop = !curSession.startFrame;
    curSession.noteHistory.push(
        {on: true, pitch: keyIndex, frame: getSessionCurrentFrame()});
    if (startLoop) {
      syncWithServer();
    }
  }
  melodySynth.triggerAttack(note, '+0', velocity);
  visual.noteOn(keyIndex);
}

/**
 * Release melody note and add to history
 * @param {string} note
 */
function releaseNote(note) {
  const keyIndex = noteToPitch[note];
  if (curSession) {
    curSession.noteHistory.push(
        {on: false, pitch: keyIndex, frame: getSessionCurrentFrame()});
  }
  melodySynth.triggerRelease(note);
  visual.noteOff(keyIndex);
}

/**
 * Schedule note to play in the future
 * @param {string} note
 * @param {boolean} on - If the note is a hit or release
 * @param {string} playTime - ToneJS time for scheduled event
 * @return {number} ID of scheduled ToneJS event
 */
function scheduleNote(note, on, playTime) {
  const keyIndex = noteToPitch[note];
  let eventID;
  if (on) {
    eventID = Tone.Transport.scheduleOnce(time => {
      chordSynth.triggerAttack(note, '+0', chordVelocity);
      Tone.Draw.schedule(() => {
        visual.noteOn(keyIndex, 'blue');
      }, time);
    }, playTime);
  } else {
    eventID = Tone.Transport.scheduleOnce(time => {
      chordSynth.triggerRelease(note);
      Tone.Draw.schedule(() => {
        visual.noteOff(keyIndex);
      }, time);
    }, playTime);
  }
  return eventID;
}

/**
 * Load sample instrument
 * @param {string} instrument
 * @return {Promise!}
 */
function loadInstrument(instrument) {
  return new Promise((resolve) => {
    const sampler = SampleLibrary.load({
      instruments: instrument,
      baseUrl: 'https://nbrosowsky.github.io/tonejs-instruments/samples/',
      onload: () => {
        resolve(sampler);
      },
    });
  });
}

/**
 * Load Salamander piano instrument
 * @return {Promise!}
 */
function loadSalamander() {
  const urls = {};
  for (const baseNote of ['A', 'C', 'Ds', 'Fs']) {
    for (const pitch of Array(7).keys()) {
      urls[`${baseNote.replace('s', '#')}${pitch + 1}`] =
          `${baseNote}${pitch + 1}.mp3`;
    }
  }
  return new Promise((resolve) => {
    const sampler = new Tone.Sampler({
      urls,
      baseUrl: 'https://tonejs.github.io/audio/salamander/',
      onload: () => {
        resolve(sampler);
      },
    });
  });
}

/**
 * Load ToneJS synth instrument
 * @param {Object!} constructor
 * @return {Promise!}
 */
function loadSynth(constructor) {
  return new Promise(resolve => resolve(new Tone.PolySynth(constructor)));
}

/**
 * Load all available instruments
 * @param {Object!} dest - ToneJS context destination
 * @return {Object!} mapping of instrument name to ToneJS instrument
 */
async function loadInstruments(dest) {
  const instrumentNames = [
    'Piano (Salamander)', 'Piano (Versilian)', 'AM Synth', 'FM Synth', 'Harp',
    'Acoustic Guitar'
  ];
  const loadFns = [
    loadSalamander(), loadInstrument('piano'), loadSynth(Tone.AMSynth),
    loadSynth(Tone.FMSynth), loadInstrument('harp'),
    loadInstrument('guitar-acoustic')
  ];
  const instruments = await Promise.all(loadFns);
  return instrumentNames.reduce(
      (prev, name, idx) =>
          ({...prev, [name]: instruments[idx].connect(dest).toDestination()}),
      {});
}

/** Create instrument dropdown options and callbacks */
function setupInstrumentSelection() {
  for (const selectEl of [chordInstSelect, melodyInstSelect]) {
    for (const name of Object.keys(instrumentMap)) {
      const option = document.createElement('option');
      option.append(name);
      option.value = name;
      selectEl.appendChild(option);
    }
  }
  chordInstSelect.addEventListener('change', event => {
    chordSynth.releaseAll();
    chordSynth = instrumentMap[event.target.value];
    chordInstSelect.blur();
  });
  melodyInstSelect.addEventListener('change', event => {
    melodySynth.releaseAll();
    melodySynth = instrumentMap[event.target.value];
    melodyInstSelect.blur();
  });
  chordInstSelect.value = DEFAULTS.chordInstrument;
  melodyInstSelect.value = DEFAULTS.melodyInstrument;
}

/** Create mouse events for note playing */
function enableClickingInputs() {
  const keys = document.querySelectorAll('rect');
  let index, note;
  keys.forEach(key => {
    const playKey = () => {
      index = key.getAttribute('data-index');
      note = pianoNotes[index];
      playNote(note, melodyVelocity);
    };

    const releaseKey = () => {
      index = key.getAttribute('data-index');
      note = pianoNotes[index];
      releaseNote(note);
    };

    key.addEventListener('mousedown', playKey);
    key.addEventListener('mouseenter', () => {
      if (mouseDown) {
        playKey();
      }
    });
    key.addEventListener('mouseup', releaseKey);
    key.addEventListener('mouseleave', releaseKey);
  });
}

/** Set computer key to note mapping and reset necessary globals */
function setKeysToNotes() {
  keysToNotes = {
    'a': `C${compKeyboardOctave}`,
    'w': `C#${compKeyboardOctave}`,
    's': `D${compKeyboardOctave}`,
    'e': `D#${compKeyboardOctave}`,
    'd': `E${compKeyboardOctave}`,
    'f': `F${compKeyboardOctave}`,
    't': `F#${compKeyboardOctave}`,
    'g': `G${compKeyboardOctave}`,
    'y': `G#${compKeyboardOctave}`,
    'h': `A${compKeyboardOctave}`,
    'u': `A#${compKeyboardOctave}`,
    'j': `B${compKeyboardOctave}`,
    'k': `C${compKeyboardOctave + 1}`,
    'o': `C#${compKeyboardOctave + 1}`,
    'l': `D${compKeyboardOctave + 1}`,
    'p': `D#${compKeyboardOctave + 1}`,
    ';': `E${compKeyboardOctave + 1}`,
    '\'': `F${compKeyboardOctave + 1}`,
  };
  visual.setComputerKeyOctave(compKeyboardOctave);
  enableClickingInputs();  // Add listeners on new keys since old ones removed
}

/** Create key events for playing notes with the computer keyboard */
function enableKeyboardInputs() {
  setKeysToNotes();
  document.addEventListener('keydown', event => {
    const note = keysToNotes[event.key];
    if (note && !heldNotes[note]) {
      heldNotes[note] = true;
      playNote(note, melodyVelocity);
    }
  });
  document.addEventListener('keyup', event => {
    const note = keysToNotes[event.key];
    if (note) {
      heldNotes[note] = false;
      releaseNote(note);
    } else if (event.key === 'z') {
      compKeyboardOctave =
          Math.max(compKeyboardOctave - 1, compKeyOctaveRange[0]);
      setKeysToNotes();
    } else if (event.key === 'x') {
      compKeyboardOctave =
          Math.min(compKeyboardOctave + 1, compKeyOctaveRange[1]);
      setKeysToNotes();
    }
  });
}

/** Create MIDI input dropdown options */
function enableMIDIInputs() {
  if (WebMidi.inputs.length < 1) {
    console.log('No MIDI device detected.');
    webMIDIInputs = [];
    interfaceSelect.style = 'display: none';
  } else {
    console.log('MIDI devices detected.');
    webMIDIInputs = WebMidi.inputs;
    webMIDIInputs.forEach(input => {
      const name = document.createTextNode(input.name);
      const option = document.createElement('option');
      option.append(name);
      option.value = input.name;
      interfaceSelect.appendChild(option);
    });
    selectMIDIInterface(webMIDIInputs[0].name);
  }
}

/**
 * Select active MIDI input device
 * @param {string} name - dropdown value associated with input
 */
function selectMIDIInterface(name) {
  interfaceSelect.value = name;

  webMIDIInputs.forEach(input => input.removeListener());
  let curInterface = webMIDIInputs.find(input => input.name === name);

  // Display the note name and play the note
  curInterface.channels[1].addListener('noteon', e => {
    playNote(e.note.identifier, e.velocity);
  });
  curInterface.channels[1].addListener('noteoff', e => {
    releaseNote(e.note.identifier);
  });
}

/**
 * Load metronome ToneJS sample instrument
 * @return {Promise!}
 */
function loadMetronome() {
  return new Promise((resolve) => {
    const metronome = SampleLibrary.load({
      instruments: 'metronome',
      baseUrl: 'https://lukewys.github.io/files/tonejs-samples/',
      onload: () => {
        resolve(metronome.toDestination());
      },
    });
  });
}

/**
 * Set ToneJS bpm
 * @param {number} bpm
 */
function setBPM(bpm) {
  Tone.Transport.bpm.value = bpm;
}

/**
 * Set ToneJS time signature
 * @param {number} timeSig
 */
function setTimeSig(timeSig) {
  Tone.Transport.timeSignature = timeSig;
  timeSigInput.value = Tone.Transport.timeSignature;
}

/** Schedule repeating events for metronome */
function startMetronome() {
  metronomeStatus = true;
  metronomeBtn.textContent = 'Disable Metronome';
  metronomeEvents.push(Tone.Transport.scheduleRepeat(time => {
    metronome.triggerAttackRelease('C1', '8n', time, metronomeVelocity);
  }, '1m', 0));
  const offBeatFreq = metronomeFreq === 'beat' ? 1 : 4;
  for (let i = 1; i < Tone.Transport.timeSignature * offBeatFreq; i++) {
    const offBeatOnset = metronomeFreq === 'beat' ? `0:${i}:0` : `0:0:${i}`;
    metronomeEvents.push(Tone.Transport.scheduleRepeat(time => {
      metronome.triggerAttackRelease('C0', '8n', time, metronomeVelocity);
    }, '1m', offBeatOnset));
  }
  if (Tone.Transport.state !== 'started') {
    Tone.Transport.start();
  }
}

/** Cancel scheduled metronome events */
function stopMetronome() {
  metronomeStatus = false;
  metronomeBtn.textContent = 'Enable Metronome';
  metronomeEvents.forEach(id => Tone.Transport.clear(id));
  metronomeEvents = [];
  if (!curSession) {
    Tone.Transport.stop();
  }
}

/** Start or stop playing metronome */
function toggleMetronome() {
  if (metronomeStatus) {
    stopMetronome();
  } else {
    startMetronome();
  }
}

/** Toggle metronome frequency */
function toggleMetronomeFreq() {
  if (metronomeFreq === 'beat') {
    metronomeFreq = 'frame';
    metronomeFreqBtn.textContent = '1/16';
  } else {
    metronomeFreq = 'beat';
    metronomeFreqBtn.textContent = '1/4';
  }
}

/**
 * Get frames since ToneJS start
 * @return {number}
 */
function getTransportFrame() {
  const [bars, beats, frames] = Tone.Transport.position.split(':');
  return parseInt(bars) * 16 + parseInt(beats) * 4 + parseFloat(frames);
}

/**
 * Convert frame to transport time format
 * @param {number} frame
 * @return {string}
 */
function frameToTransportTime(frame) {
  const bars = Math.floor(frame / 16);
  const beats = Math.floor((frame % 16) / 4);
  const frames = frame % 4;
  return `${bars}:${beats}:${frames}`;
}

/**
 * Initialize globals, load everything, setup input handlers
 * @param {NoteVisual!} visual_arg - initialized note visualizer class
 */
async function initializeMIDIReader(visual_arg) {
  visual = visual_arg;
  Tone.Transport.timeSignature = DEFAULTS.timeSignature;
  Tone.Transport.bpm.value = DEFAULTS.bpm;

  // Setup audio recorder
  const actx = Tone.context;
  const dest = actx.createMediaStreamDestination();
  recorder = new MediaRecorder(dest.stream);
  recorder.ondataavailable = (e) => {
    curAudioRecording.push(e.data);
  };
  recorder.onstop = saveSessionRecording;

  // Load instruments and metronome
  instrumentMap = await loadInstruments(dest);
  chordSynth = instrumentMap[DEFAULTS.chordInstrument];
  melodySynth = instrumentMap[DEFAULTS.melodyInstrument];
  metronome = await loadMetronome();

  // Set up button and input handlers
  playBtn = document.getElementById('play-btn');
  playBtn.addEventListener('click', showMainScreen);
  metronomeBtn = document.getElementById('metronome-button');
  metronomeBtn.addEventListener('click', toggleMetronome);
  bpmInput = document.getElementById('bpm-input');
  bpmInput.value = Tone.Transport.bpm.value;
  addNumericInputEventListener(bpmInput, setBPM);
  timeSigInput = document.getElementById('time-sig-input');
  timeSigInput.value = DEFAULTS.timeSignature;
  addNumericInputEventListener(timeSigInput, setTimeSig);
  metronomeFreqBtn = document.getElementById('metronome-freq-button');
  metronomeFreqBtn.addEventListener('click', toggleMetronomeFreq);
  if (window.location.hostname !== 'localhost') {
    metronomeFreqBtn.style = 'display: none';
  }
  interfaceSelect = document.getElementById('interface-select');
  interfaceSelect.addEventListener(
      'change', event => selectMIDIInterface(event.target.value));
  modelSelect = document.getElementById('model-select');
  liveSessionBtn = document.getElementById('live-session-button');
  liveSessionBtn.addEventListener('click', toggleLiveSession);
  downloadSessionCheck = document.getElementById('download-session-check');
  metronomeCheck = document.getElementById('metronome-check');
  showChordsCheck = document.getElementById('show-chords-check');
  temperatureInput = document.getElementById('temperature-input');
  temperatureInput.value = DEFAULTS.temperature;
  addNumericInputEventListener(temperatureInput);
  silenceInput = document.getElementById('silence-input');
  silenceInput.value = DEFAULTS.silence;
  addNumericInputEventListener(silenceInput);
  commitaheadInput = document.getElementById('commitahead-input');
  commitaheadInput.value = DEFAULTS.commitahead;
  commitaheadInput.max = DEFAULTS.lookahead;
  addNumericInputEventListener(commitaheadInput);
  lookaheadInput = document.getElementById('lookahead-input');
  lookaheadInput.value = DEFAULTS.lookahead;
  setLookahead(DEFAULTS.lookahead);
  addNumericInputEventListener(lookaheadInput, setLookahead);
  chordInstSelect = document.getElementById('chord-inst-select');
  melodyInstSelect = document.getElementById('melody-inst-select');
  setupInstrumentSelection();

  // Do initial setup with server
  establishServerConnection();

  console.log('ready!');
  playBtn.textContent = 'Play';
  playBtn.removeAttribute('disabled');
  playBtn.classList.remove('loading');
}
